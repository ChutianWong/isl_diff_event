"""Cut one APS frame and its event window from an AEDAT4 recording into an acquisition directory.

    python tools/make_window.py RECORDING.aedat4 data/<group>/<name> [--start-us 3.3e6] [--window-us 100000]

Writes raw/window.npz (the frame and the events), raw/window.json (name and SHA-256 of the
recording, the selected frame) and config.yaml (the measurement facts the loader checks), so
that the directory runs with

    python -m neurosr data/<group>/<name> --output results/<name>

The frame is the last one whose exposure starts before --start-us, counted in microseconds from
the first event of the recording.  The events run from that exposure start over --window-us, and
their times restart at the first of them.  The defaults reproduce the bundled window from
holography/recording_001.aedat4.

Reading AEDAT4 needs the separately installed dv package (AGPLv3); nothing of dv is included
here, and reconstruction itself reads only the NPZ.  Colour recordings also need opencv-python.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CONFIG = """# Measurement facts only, in SI units.
dataset:
  name: {name}
data:
  file: raw/window.npz
  sha256: {sha256}
  layout: frame[y,x]; events xs,ys,ts,ps; no rescaling
sensor:
  shape_yx:
  - {height}
  - {width}
optics:
  wavelength_m: {wavelength_m!r}
  pixel_pitch_m: {pixel_pitch_m!r}
acquisition:
  window_start_us: {start}
  exposure_end_us: {end}
  window_us: {win}
  frame_index: {frame_ind}
  exposure_us: {win_frame}
  event_count: {events}
  event_time_origin: first selected event; microseconds
  polarity: 0=negative, 1=positive
  frame_scale: uint8, 0 to 255
"""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def time_window(xs, ys, ts, ps, s, win):
    """The events with s <= t <= s + win."""
    loc = np.where((ts <= s + win) & (ts >= s))
    ts, xs, ys, ps = ts[loc], xs[loc], ys[loc], ps[loc]
    return xs, ys, ts, ps


def load_events(path):
    """Events (t, x, y, p), APS frames and their exposure intervals; times in microseconds from
    the first event of the recording."""
    try:
        from dv import AedatFile
    except ImportError as error:
        raise SystemExit("Reading AEDAT4 needs the dv package: python -m pip install dv") from error
    with AedatFile(path) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])
        frames = np.stack([packet for packet in f['frames']])
    ts, xs, ys, ps = events['timestamp'], events['x'], events['y'], events['polarity']
    t_ref = ts.min()
    ts = ts - t_ref
    frames_img = []
    time_img = []
    for i in range(frames.size):
        frames_img.append(frames[i].image)
        time_img.append([frames[i].timestamp_start_of_exposure, frames[i].timestamp_end_of_exposure])
    frames_img = np.asarray(frames_img).squeeze()
    time_img = np.asarray(time_img).squeeze()
    time_img = time_img - t_ref
    return ts, xs, ys, ps, frames_img, time_img


def load_window(path, s, win):
    """One frame and its event window, in microseconds from the first event of the recording.

    The frame is the last one whose exposure starts before s.  The events run from its exposure
    start over win and restart at 0; first_event_offset is the time of the first of them after
    the exposure start.  frames are grey [F, H, W] in 0-255 and frame = frames[frame_ind].
    """
    ts, xs, ys, ps, frames, time_img = load_events(path)
    ts = ts - ts.min()
    if time_img.ndim != 2 or len(time_img) < 2:
        raise ValueError("The recording needs at least two APS frames")
    if frames.ndim == 4:
        import cv2
        frames = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    later = np.where(time_img[:, 0] > s)[0]
    if len(later) == 0 or later[0] == 0:
        raise ValueError(f"--start-us must lie after the first APS exposure start ({int(time_img[0, 0])} us) "
                         f"and before the last one ({int(time_img[-1, 0])} us)")
    frame_ind = int(later[0] - 1)
    start, end = time_img[frame_ind]
    win_frame = int(end - start)
    win = int(win)
    if start < 0:
        raise ValueError(f"The selected exposure starts {int(-start)} us before the first event; "
                         "choose a later --start-us")
    if win_frame > win:
        raise ValueError(f"The exposure ({win_frame} us) is longer than --window-us ({win} us)")
    if start + win > ts[-1]:
        raise ValueError(f"The recording ends {int(start + win - ts[-1])} us before the window does")
    xs, ys, ts, ps = time_window(xs, ys, ts, ps, s=start, win=win)
    if len(ts) == 0:
        raise ValueError("No events in the selected window")
    offset = int(ts.min() - start)
    ts = ts - ts.min()
    return SimpleNamespace(xs=xs, ys=ys, ts=ts, ps=ps, frames=frames, time_img=time_img, frame_ind=frame_ind,
                           frame=frames[frame_ind], start=int(start), end=int(end), win=win, win_frame=win_frame,
                           first_event_offset=offset)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path, help="AEDAT4 recording")
    parser.add_argument("acquisition", type=Path, help="new acquisition directory, e.g. data/<group>/<name>")
    parser.add_argument("--start-us", type=float, default=3.3e6,
                        help="select the last exposure starting before this time (default 3.3e6)")
    parser.add_argument("--window-us", type=int, default=100000, help="event window length (default 100000)")
    parser.add_argument("--wavelength-m", type=float, default=632.8 * 1e-9, help="default 632.8 nm")
    parser.add_argument("--pixel-pitch-m", type=float, default=18.5 * 1e-6, help="default 18.5 um (DAVIS346)")
    args = parser.parse_args()
    if not args.recording.is_file():
        parser.error(f"No recording at {args.recording}")
    if args.window_us <= 0 or args.wavelength_m <= 0 or args.pixel_pitch_m <= 0:
        parser.error("--window-us, --wavelength-m and --pixel-pitch-m must be positive")
    output = args.acquisition / "raw" / "window.npz"
    config = args.acquisition / "config.yaml"
    if args.acquisition.exists() and not args.acquisition.is_dir():
        parser.error(f"{args.acquisition} is a file; name a directory")
    if output.exists() or config.exists():
        parser.error(f"{args.acquisition} already holds a window; choose a new acquisition directory")
    try:
        rec = load_window(str(args.recording), args.start_us, args.window_us)
    except ValueError as error:
        parser.error(str(error))
    except Exception as error:   # dv raises several types on a file that is not AEDAT4
        parser.error(f"Cannot read {args.recording} as AEDAT4 ({type(error).__name__}: {error})")
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = ("xs", "ys", "ts", "ps", "frame", "frame_ind", "win_frame", "win", "start", "end")
    np.savez_compressed(output, **{k: np.asarray(getattr(rec, k)) for k in keys})
    report = dict(source_recording=args.recording.name, source_sha256=sha256(args.recording),
                  start_us=rec.start, end_us=rec.end, frame_index=rec.frame_ind,
                  exposure_us=rec.win_frame, events=len(rec.ts), first_event_offset_us=rec.first_event_offset)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    resolved = args.acquisition.resolve()
    config.write_text(CONFIG.format(name=f"{resolved.parent.name}/{resolved.name}", sha256=sha256(output),
                                    height=rec.frame.shape[0], width=rec.frame.shape[1],
                                    wavelength_m=args.wavelength_m, pixel_pitch_m=args.pixel_pitch_m,
                                    start=rec.start, end=rec.end, win=rec.win, frame_ind=rec.frame_ind,
                                    win_frame=rec.win_frame, events=len(rec.ts)),
                      encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.acquisition.as_posix()}; run: python -m neurosr {args.acquisition.as_posix()} "
          f"--output results/{resolved.name}")


if __name__ == "__main__":
    main()
