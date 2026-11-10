"""Offline aedat4 conversion with the original reader and load_window.

Requires optional dv (AGPLv3); no dv code is vendored. Reconstruction uses only NPZ.
For colour recordings load_window also needs optional opencv-python.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from dv import AedatFile

def  time_window(xs, ys,ts, ps ,s , win ):
    loc = np.where((ts <= s + win) & (ts >= s))
    ts, xs, ys, ps = ts[loc], xs[loc], ys[loc], ps[loc]
    return xs, ys, ts, ps


def load_events(eve_dtype="aedat4", path=None):
    with AedatFile(path) as f:
        # events will be a named numpy array
        events = np.hstack([packet for packet in f['events'].numpy()])
        # frames_img = np.stack([packet.image for packet in f['frames']], axis=3)
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


def load_window(path, s, win, compare_2frame=False):
    """The demos' window on an aedat4 recording, in microseconds from the first event.

    The frame is the last one whose exposure starts before s.  The events run from its exposure
    start over max(win, exposure), or to the end of the next exposure with compare_2frame, and
    restart at 0.  frames are grey [F, H, W] in 0-255, frame = frames[frame_ind].
    """
    ts, xs, ys, ps, frames, time_img = load_events(eve_dtype="aedat4", path=path)
    ts = ts - ts.min()
    if frames.ndim == 4:
        import cv2
        frames = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    frame_ind = int(np.where(time_img[:, 0] > s)[0][0] - 1)
    start, end = time_img[frame_ind]
    win_frame = int(end - start)
    if compare_2frame and frame_ind + 1 < len(time_img):
        win = int(time_img[frame_ind + 1][1] - start)
    win = max(int(win), win_frame)
    xs, ys, ts, ps = time_window(xs, ys, ts, ps, s=start, win=win)
    ts = ts - ts.min()
    return SimpleNamespace(xs=xs, ys=ys, ts=ts, ps=ps, frames=frames, time_img=time_img, frame_ind=frame_ind,
                           frame=frames[frame_ind], start=int(start), end=int(end), win=win, win_frame=win_frame)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-us", type=float, default=3.3e6)
    parser.add_argument("--window-us", type=int, default=100000)
    args = parser.parse_args()
    rec = load_window(str(args.recording), args.start_us, args.window_us)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    keys = ("xs", "ys", "ts", "ps", "frame", "frame_ind", "win_frame", "win", "start", "end")
    np.savez_compressed(args.output, **{k: np.asarray(getattr(rec, k)) for k in keys})
    report = dict(source_recording=args.recording.name,
                  source_sha256=hashlib.sha256(args.recording.read_bytes()).hexdigest(),
                  start_us=rec.start, end_us=rec.end, frame_index=rec.frame_ind,
                  exposure_us=rec.win_frame, events=len(rec.ts))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
