"""Extract the two APS exposures and events for a joint NeuroSR preset."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from make_window import load_events, time_window

FRAME_INDICES = {"earthworm": 7, "testis": 69}
WINDOW_US = 100_000


def extract(recording, scene):
    ts, xs, ys, ps, frames, exposures = load_events(path=str(recording))
    index = FRAME_INDICES[scene]
    if exposures.ndim != 2 or index + 1 >= len(exposures):
        raise ValueError("The preset needs the selected APS frame and the next exposure")
    start = int(exposures[index, 0])
    intervals = exposures[index:index + 2] - start
    if not 0 < intervals[0, 1] <= intervals[1, 0] < intervals[1, 1] <= WINDOW_US:
        raise ValueError("Both consecutive APS exposures must fit in the 100 ms window")
    if ts.min() > start or ts.max() < start + WINDOW_US:
        raise ValueError("Recording does not cover the complete 100 ms window")
    x, y, t, p = time_window(xs, ys, ts, ps, s=start, win=WINDOW_US)
    if len(t) == 0 or np.any(np.diff(t) < 0):
        raise ValueError("Expected nonempty, sorted window events")
    selected = frames[index:index + 2]
    if selected.ndim == 4:
        import cv2

        selected = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in selected])
    digest = hashlib.sha256()
    with recording.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    metadata = dict(scene=scene, recording_sha256=digest.hexdigest(), frame_index=index,
                    exposure_start_us=start, window_us=WINDOW_US)
    return dict(xs=x, ys=y, ts=t - start, ps=p, frames=selected, exposures_us=intervals,
                metadata=json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=FRAME_INDICES)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz" or args.output.exists():
        parser.error("Supply a new .npz output file")
    arrays = extract(args.recording, args.scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Saved {len(arrays['ts'])} events and two APS frames to {args.output}")


if __name__ == "__main__":
    main()
