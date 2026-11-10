"""Run joint NeuroSR from an exposure-aligned two-frame window."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.optimize import minimize_scalar

from ._constants import CONTRAST_0
from .events import EventModel
from .frame import FrameModel, MultiFrameEDI
from .joint import PIXEL_PITCH_M, WAVELENGTH_M, reconstruct_joint
from .model import NeuroSR
from .motion import SplineFlow, register_segments
from .ops import AdamP, C_MAX, C_MIN, np_to_torch, raw_from_bounded_softplus, time_window
from .optics import AngularSpectrum

PRESETS = {"earthworm": (7, 3), "testis": (69, 2)}
WINDOW_US = 100_000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_contrast(medi):
    """Fit one effective c from the two exposure centres."""
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


def prepare(window, device):
    with np.load(window, allow_pickle=False) as data:
        info = json.loads(data["metadata"].item())
        xs, ys, ts, ps = (data[key] for key in ("xs", "ys", "ts", "ps"))
        selected = data["frames"]
        intervals = data["exposures_us"]
    scene = info["scene"]
    frame_index, scale = PRESETS[scene]
    holo = scene == "testis"
    if info["frame_index"] != frame_index or info["window_us"] != WINDOW_US:
        raise ValueError("Window metadata does not match the preset")
    if (selected.ndim != 3 or selected.shape[0] != 2 or intervals.shape != (2, 2)
            or intervals[0, 0] != 0
            or not 0 < intervals[0, 1] <= intervals[1, 0] < intervals[1, 1] <= WINDOW_US):
        raise ValueError("Expected two consecutive APS exposures on the selected window clock")
    if (ts.ndim != 1 or len(ts) == 0 or not (xs.shape == ys.shape == ts.shape == ps.shape)
            or ts[0] < 0 or ts[-1] > WINDOW_US or np.any(np.diff(ts) < 0)):
        raise ValueError("Expected matching event arrays sorted on the APS window clock")
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

    exposure_us = int(intervals[0, 1])
    frame_scale = float(selected[0].max()) if holo else 255.
    if frame_scale <= 0:
        raise ValueError("The selected hologram has zero intensity")
    frame = torch.from_numpy(selected[0] / frame_scale).to(device)
    edi_frame = FrameModel(frame, xs, ys, ts, ps, exposure_us, raw_c, sensor, align_exposure=True)
    with torch.no_grad():
        edi = edi_frame.latent()
    print(f"Frame {frame_index}: {len(ts)} events, effective c = {c:.8g}", flush=True)

    # Retain the validated CMax registration clock. The joint terms below
    # use the APS clock, including the fractional exposure centre.
    motion_times = ts - ts[0]
    flow = SplineFlow(WINDOW_US, 12, sensor, device=device)
    flow.set_segments(register_segments(xs, ys, motion_times, ps, flow))
    motion_events = EventModel(*np_to_torch(xs, ys, motion_times, ps, device), sensor,
                               fidelity="normalized" if holo else "contrast")
    seed, distance = None, None
    report = dict(contrast=c, events=len(ts), reference_us=exposure_us / 2,
                  first_event_offset_us=int(ts[0]),
                  exposure_intervals_us=intervals.tolist(), frame_scale=frame_scale,
                  calibration=calibration)
    if holo:
        optical = AngularSpectrum(WAVELENGTH_M, PIXEL_PITCH_M, sensor, device=device, learn_distance=True)
        recorded = torch.from_numpy(selected[0]).to(device)
        z_af, _ = optical.autofocus(recorded, np.linspace(.015, .021, 100))
        sharp = FrameModel(recorded.double() / frame_scale, xs, ys, motion_times, ps, exposure_us,
                           motion_events.raw_c, sensor, edi=False, domain="amplitude")
        initializer = NeuroSR(flow, motion_events, sharp, optical, optimizer=AdamP,
                              kappa=dict(event=0., bg=0., tv_sr=0., tv_object=.04, laplacian=.001),
                              lr=dict(background=0., motion=0., distance=1e-5),
                              schedule=dict(joint=(500, .9)))
        initializer.cmax(log_every=700)
        backpropagated = optical((recorded / frame_scale).sqrt(), -z_af).to(torch.complex64)
        initializer.joint(backpropagated, iterations=4001, log_every=500)
        seed = initializer.object_m1
        distance = float(optical.distance.detach())
        report.update(autofocus_z_m=float(z_af), initial_z_m=distance)
    else:
        NeuroSR(flow, motion_events, edi_frame).cmax(log_every=700)
    flow.requires_grad_(False)
    events = tuple(torch.as_tensor(values, dtype=torch.float32, device=device) for values in (xs, ys, ts, ps))
    return edi, events, flow, frame, seed, distance, dict(info, **report), scale


def save_figure(path, rows):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(4 * len(rows[0]), 3.3 * len(rows)),
                             squeeze=False, layout="constrained")
    for row, panels in zip(axes, rows):
        for ax, (image, title, cmap, limits) in zip(row, panels):
            ax.imshow(image, cmap=cmap, vmin=limits[0], vmax=limits[1])
            ax.set_title(title)
            ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_figures(out, arrays, scale):
    if "object_sr" in arrays:
        fields = (arrays["object_m1"], arrays["object_sr"])
        titles = ("Frame-only M=1", f"Joint M={scale}")
        rows = [[(abs(field), title + " amplitude", "gray", (0, 1))
                 for field, title in zip(fields, titles)],
                [(np.angle(field * field.mean().conj()), title + " phase (rad)", "viridis", (-1, 1))
                 for field, title in zip(fields, titles)]]
    else:
        rows = [[(arrays[key], title, "gray", (0, 1)) for key, title in (
            ("frame", "APS frame"), ("edi", "EDI"), ("joint", f"Joint + log-TV M={scale}"))]]
    save_figure(out / "reconstruction.png", rows)
    iwe = [(arrays[key] * arrays["iwe_support"]).sum(0)
           for key in ("iwe_measured", "iwe_predicted", "iwe_residual")]
    limit = max(float(np.percentile(np.abs(np.stack(iwe)), 99)), np.finfo(np.float32).eps)
    save_figure(out / "iwe.png", [[(image, title, "seismic", (-limit, limit))
                                  for image, title in zip(iwe, ("Measured c × IWE", "Predicted", "Predicted − measured"))]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path, help="NPZ from tools/make_joint_window.py")
    parser.add_argument("--output", type=Path, required=True, help="New directory for arrays, figures and metadata")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not args.window.is_file() or args.output.exists():
        parser.error("Supply an existing window NPZ and a new output directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu for a slower run.")
    matplotlib.use("Agg")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    edi, events, flow, frame, seed, distance, report, scale = prepare(args.window, device)
    print(f"Fitting joint event terms at M={scale}", flush=True)
    arrays, fit = reconstruct_joint(edi, events, flow, contrast=report["contrast"],
                                    reference_us=report["reference_us"], scale=scale,
                                    object_seed=seed, distance_m=distance)
    arrays.update(frame=frame, segments_px=flow.segments())
    arrays = {key: tensor.detach().cpu().numpy() for key, tensor in arrays.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("A result array is nonfinite")
    sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    report.update(fit, M=scale, window_sha256=sha256(args.window),
                  source_sha256={source.name: sha256(source) for source in sources})
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "result.npz", **arrays)
    save_figures(args.output, arrays, scale)
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
