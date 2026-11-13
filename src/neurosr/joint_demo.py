"""Joint NeuroSR from a two-exposure window: calibration, transport, anchor, reconstruction.

    1. contrast   mEDI between the two consecutive exposures gives one effective threshold c
    2. anchor     EDI refers the first exposure to its centre
    3. transport  contrast maximisation of the events gives the piecewise-linear path
    4. latent     the joint objective at the native scale, then at M (neurosr.joint)

Holography first fits the complex field and the propagation distance from the frame alone, then
refines both jointly at M.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.optimize import minimize_scalar

from ._constants import CONTRAST_0, PIXEL_PITCH_M, WAVELENGTH_M
from .data import output_directory
from .events import EventModel
from .frame import FrameModel, MultiFrameEDI
from .joint import reconstruct_joint
from .model import NeuroSR
from .motion import SplineFlow, register_segments
from .ops import AdamP, C_MAX, C_MIN, np_to_torch, raw_from_bounded_softplus, time_window
from .optics import AngularSpectrum

# scene: (zero-based index of the first APS frame, M, whether the latent is a complex field)
PRESETS = {"earthworm": (7, 3, False), "testis": (69, 2, True)}
WINDOW_US = 100_000
SEGMENTS = 12


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_contrast(medi):
    """Fit one effective c from the two exposure centres: a grid, then a bounded search."""
    if not bool((medi.pos_transfer + medi.neg_transfer).any()):
        raise ValueError("No events between the exposure centres for mEDI")
    candidates = []

    @torch.no_grad()
    def objective(c):
        medi.raw_c.copy_(raw_from_bounded_softplus(c).to(medi.raw_c))
        value = float(medi.loss())
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite mEDI loss at c={c:g}")
        candidates.append((value, float(c)))
        return value

    initial = objective(CONTRAST_0)
    grid = np.linspace(C_MIN + 1e-5, C_MAX - 1e-5, 21)
    best = int(np.argmin([objective(float(c)) for c in grid]))
    bracket = (float(grid[max(0, best - 1)]), float(grid[min(len(grid) - 1, best + 1)]))
    search = minimize_scalar(objective, bounds=bracket, method="bounded",
                             options={"xatol": 1e-6, "maxiter": 60})
    final, c = min(candidates, key=lambda item: item[0])
    return c, dict(initial_loss=initial, final_loss=final, search_success=bool(search.success),
                   inter_centre_events=int((medi.pos_transfer + medi.neg_transfer).sum()))


def prepare(window, device, cmax_steps=2100, field_steps=4001):
    """Everything the joint objective holds fixed: the anchor, the events on the APS clock, the
    transport and c; for holography also the frame-only field and distance.  cmax_steps and
    field_steps are the update counts of the transport fit and of the frame-only field fit."""
    with np.load(window, allow_pickle=False) as data:
        info = json.loads(data["metadata"].item())
        xs, ys, ts, ps = (data[key] for key in ("xs", "ys", "ts", "ps"))
        selected = data["frames"]
        intervals = data["exposures_us"]
    scene = info["scene"]
    if scene not in PRESETS:
        raise ValueError(f"No preset for scene {scene!r}; the presets are {', '.join(PRESETS)}")
    frame_index, scale, holo = PRESETS[scene]
    if info["frame_index"] != frame_index or info["window_us"] != WINDOW_US:
        raise ValueError("Window metadata does not match the preset")
    if (selected.ndim != 3 or selected.shape[0] != 2 or intervals.shape != (2, 2)
            or intervals[0, 0] != 0
            or not 0 < intervals[0, 1] <= intervals[1, 0] < intervals[1, 1] <= WINDOW_US):
        raise ValueError("Expected two consecutive APS exposures on the selected window clock")
    if (ts.ndim != 1 or len(ts) == 0 or not (xs.shape == ys.shape == ts.shape == ps.shape)
            or ts[0] < 0 or ts[-1] > WINDOW_US or np.any(np.diff(ts) < 0)):
        raise ValueError("Expected matching event arrays sorted on the APS window clock")

    # 1. contrast threshold from the transfer between the two exposure centres
    xc, yc, tc, pc = time_window(xs, ys, ts, ps, s=0, win=int(intervals[-1, 1]))
    selected = selected.astype(np.float32)
    sensor = tuple(selected.shape[-2:])
    raw_c = torch.nn.Parameter(raw_from_bounded_softplus(CONTRAST_0).to(device))
    medi = MultiFrameEDI(torch.from_numpy(selected / 255.).to(device), intervals, xc, yc, tc, pc, raw_c)
    c, calibration = fit_contrast(medi)
    del medi
    with torch.no_grad():
        raw_c.copy_(raw_from_bounded_softplus(c).to(raw_c))
    raw_c.requires_grad_(False)

    # 2. photometric anchor: the first exposure referred to its centre
    exposure_us = int(intervals[0, 1])
    frame_scale = float(selected[0].max()) if holo else 255.
    if frame_scale <= 0:
        raise ValueError("The selected hologram has zero intensity")
    frame = torch.from_numpy(selected[0] / frame_scale).to(device)
    edi_frame = FrameModel(frame, xs, ys, ts, ps, exposure_us, raw_c, sensor, align_exposure=True)
    with torch.no_grad():
        edi = edi_frame.latent()
    print(f"Frame {frame_index}: {len(ts)} events, effective c = {c:.8g}", flush=True)

    # 3. transport by contrast maximisation, on a clock that starts at the first event.
    # The joint terms use the APS clock instead, including the fractional exposure centre.
    motion_times = ts - ts[0]
    flow = SplineFlow(WINDOW_US, SEGMENTS, sensor, device=device)
    flow.set_segments(register_segments(xs, ys, motion_times, ps, flow))
    motion_events = EventModel(*np_to_torch(xs, ys, motion_times, ps, device), sensor,
                               fidelity="normalized" if holo else "contrast")
    seed, distance = None, None
    report = dict(contrast=c, events=len(ts), reference_us=exposure_us / 2,
                  first_event_offset_us=int(ts[0]),
                  exposure_intervals_us=intervals.tolist(), frame_scale=frame_scale,
                  calibration=calibration)
    if holo:
        # frame-only complex field and propagation distance at the native scale
        optical = AngularSpectrum(WAVELENGTH_M, PIXEL_PITCH_M, sensor, device=device, learn_distance=True)
        recorded = torch.from_numpy(selected[0]).to(device)
        z_af, _ = optical.autofocus(recorded, np.linspace(.015, .021, 100))
        sharp = FrameModel(recorded.double() / frame_scale, xs, ys, motion_times, ps, exposure_us,
                           motion_events.raw_c, sensor, edi=False, domain="amplitude")
        initializer = NeuroSR(flow, motion_events, sharp, optical, optimizer=AdamP,
                              kappa=dict(event=0., bg=0., tv_sr=0., tv_object=.04, laplacian=.001),
                              lr=dict(background=0., motion=0., distance=1e-5),
                              schedule=dict(joint=(500, .9)))
        initializer.cmax(iterations=cmax_steps, log_every=700)
        backpropagated = optical((recorded / frame_scale).sqrt(), -z_af).to(torch.complex64)
        initializer.joint(backpropagated, iterations=field_steps, log_every=500)
        seed = initializer.object_m1
        distance = float(optical.distance.detach())
        report.update(autofocus_z_m=float(z_af), initial_z_m=distance)
    else:
        NeuroSR(flow, motion_events, edi_frame).cmax(iterations=cmax_steps, log_every=700)
    flow.requires_grad_(False)
    events = tuple(torch.as_tensor(values, dtype=torch.float32, device=device) for values in (xs, ys, ts, ps))
    return edi, events, flow, frame, seed, distance, dict(info, **report), scale


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m neurosr joint", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("window", type=Path, help="two-exposure window NPZ: data/joint/earthworm.npz, "
                                                  "data/joint/testis.npz, or one cut by "
                                                  "tools/make_joint_window.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory outside data/ for arrays, figures and metadata")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4, help="CPU worker threads")
    parser.add_argument("--tiny", action="store_true", help="5 iterations per stage: a smoke check, not a result")
    args = parser.parse_args(argv)
    if not args.window.is_file():
        parser.error(f"No window at {args.window}")
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        output = output_directory(args.output)
        device = torch.device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if output.exists():
        parser.error(f"{output} exists; choose a new output directory")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu for a slower run.")
    matplotlib.use("Agg")
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    start = time.time()
    short = dict(cmax_steps=5, field_steps=5) if args.tiny else {}
    edi, events, flow, frame, seed, distance, report, scale = prepare(args.window, device, **short)
    print(f"Fitting joint event terms at M={scale}", flush=True)
    arrays, fit = reconstruct_joint(edi, events, flow, contrast=report["contrast"],
                                    reference_us=report["reference_us"], scale=scale,
                                    object_seed=seed, distance_m=distance,
                                    **(dict(native_steps=5, sr_steps=5) if args.tiny else {}))
    arrays.update(frame=frame, segments_px=flow.segments())
    arrays = {key: tensor.detach().cpu().numpy() for key, tensor in arrays.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("A result array is nonfinite")
    sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    report.update(fit, M=scale, tiny=args.tiny, seconds=round(time.time() - start, 1),
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   torch=torch.__version__, device=str(device),
                                   cpu_threads=torch.get_num_threads()),
                  window_sha256=sha256(args.window),
                  source_sha256={source.name: sha256(source) for source in sources})
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "result.npz", **arrays)
    from .visualize import save_joint_figures
    save_joint_figures(output, arrays, scale)
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8", newline="\n")
    print(f"Saved {output} in {report['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
