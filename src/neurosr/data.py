"""Load a pre-extracted window without changing its dtype, order or intensity scale."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np

KEYS = ("xs", "ys", "ts", "ps", "frame", "frame_ind", "win_frame", "win", "start", "end")


def load_window_npz(config):
    """Return x,y,t,p arrays and the uint8 exposure; t is zeroed at the first event."""
    facts = config["measurement"]
    directory = config["dataset_dir"]
    path = (directory / facts["data"]["file"]).resolve()
    if directory not in path.parents:
        raise ValueError("Measurement file must be inside the acquisition directory")
    expected_hash = facts["data"].get("sha256")
    if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Measurement file SHA-256 differs from config.yaml")
    with np.load(path, allow_pickle=False) as stored:
        values = {key: stored[key] for key in KEYS}
    for key, dtype in (("xs", "int16"), ("ys", "int16"), ("ts", "int64"), ("ps", "int8"), ("frame", "uint8")):
        if values[key].dtype != np.dtype(dtype):
            raise ValueError(f"{key} must preserve original dtype {dtype}")
    for key in KEYS[5:]:
        if values[key].shape != ():
            raise ValueError(f"{key} must be scalar")
        values[key] = int(values[key])
    n = len(values["ts"])
    if n == 0 or any(values[k].shape != (n,) for k in ("xs", "ys", "ts", "ps")):
        raise ValueError("Event arrays must be nonempty, one-dimensional and equal in length")
    shape = tuple(facts["sensor"]["shape_yx"])
    if values["frame"].shape != shape:
        raise ValueError("Frame shape differs from sensor.shape_yx")
    if not np.all((values["xs"] >= 0) & (values["xs"] < shape[1])) or not np.all((values["ys"] >= 0) & (values["ys"] < shape[0])):
        raise ValueError("Event coordinates are outside the sensor")
    if values["ts"][0] != 0 or np.any(np.diff(values["ts"]) < 0) or values["ts"][-1] > values["win"]:
        raise ValueError("Event timestamps must be ordered, window-relative microseconds")
    if not np.all(np.isin(values["ps"], [0, 1])):
        raise ValueError("Stored polarity must use the original 0/1 convention")
    acquisition = facts["acquisition"]
    expected = {"frame_ind": acquisition["frame_index"], "win_frame": acquisition["exposure_us"],
                "win": acquisition["window_us"], "start": acquisition["window_start_us"], "end": acquisition["exposure_end_us"]}
    if any(values[k] != v for k, v in expected.items()) or n != acquisition["event_count"]:
        raise ValueError("Window metadata differs from acquisition facts")
    if values["end"] - values["start"] != values["win_frame"]:
        raise ValueError("Exposure duration differs from its endpoints")
    return SimpleNamespace(**values, source_path=path)
