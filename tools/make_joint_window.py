"""Cut the two consecutive APS exposures and the 100 ms event window of a joint preset.

    python tools/make_joint_window.py {earthworm,testis} RECORDING.aedat4 OUT.npz

Each preset is defined on one recording of the public dataset
(https://doi.org/10.6084/m9.figshare.33545614), listed in RECORDINGS below; the recording is
checked by its SHA-256.  The APS frame of a preset is the one the solver expects
(neurosr.joint_demo.PRESETS).  The window starts at the exposure start of that frame and the
event times are microseconds on that clock.  The two windows this writes are the ones bundled
under data/joint.

Reading AEDAT4 needs the separately installed dv package (AGPLv3); the earthworm recording is a
colour recording and also needs opencv-python.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from make_window import load_events, sha256, time_window

try:
    from neurosr.joint_demo import PRESETS, WINDOW_US
except ImportError as error:
    raise SystemExit("Install the package first: python -m pip install -e .") from error

# scene: (archive, recording in the archive, SHA-256 of the recording)
RECORDINGS = {
    "earthworm": ("01_microscopy.zip", "microscopy_01/recording_012.aedat4",
                  "24b9f090f8611fcac9d38e0755f4be013113f15e6a66cbbf49777ab97253f12f"),
    "testis": ("03_holography.zip", "holography/recording_003.aedat4",
               "3c75122ec4a43242bb44362dba928eefe452767f8a25cf463ffb38b57ca12362"),
}


def extract(recording, scene):
    archive, name, expected = RECORDINGS[scene]
    index = PRESETS[scene][0]
    digest = sha256(recording)
    if digest != expected:
        raise ValueError(f"The {scene} preset is defined on {name} of {archive}; "
                         f"{recording.name} has a different SHA-256")
    ts, xs, ys, ps, frames, exposures = load_events(str(recording))
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
    metadata = dict(scene=scene, archive=archive, recording=name, recording_sha256=digest, frame_index=index,
                    exposure_start_us=start, window_us=WINDOW_US)
    return dict(xs=x, ys=y, ts=t - start, ps=p, frames=selected, exposures_us=intervals,
                metadata=json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", choices=RECORDINGS)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.recording.is_file():
        parser.error(f"No recording at {args.recording}")
    if args.output.suffix.lower() != ".npz" or args.output.exists():
        parser.error("Supply a new .npz output file")
    try:
        arrays = extract(args.recording, args.scene)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Saved {len(arrays['ts'])} events and two APS frames to {args.output}")


if __name__ == "__main__":
    main()
