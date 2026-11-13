"""Measurement facts (config.yaml) and solver settings (config_neurosr.yaml) of an acquisition directory."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

DEFAULTS = {
    "pieces": 12, "M": 2, "seed": 0, "sigma": 0.849,
    "iterations": {"cmax": 2100, "joint": 4001, "sr": 2001},
    "autofocus": {"min_m": 0.015, "max_m": 0.021, "samples": 100},
    "kappa": {"event": 0.0, "frame": 1.0, "sog": 1e-1, "bg": 0.0, "tv": 0.0,
              "tv_sr": 0.0, "tv_object": 4e-2, "laplacian": 1e-3, "event_sr": None},
    "lr": {"cmax": 1e-2, "image": 4e-3, "background": 0.0, "motion": 0.0,
           "contrast": 1e-3, "distance": 1e-5, "distance_sr": 2e-6, "sr": 1e-2},
    "schedule": {"cmax": [200, 0.9], "joint": [500, 0.9], "sr": [500, 1.0]},
}


def read_yaml(path):
    """Read a YAML mapping with no custom constructors."""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return value


def merge_settings(defaults, overrides):
    """Merge overrides into defaults, key by key; an unknown key is an error."""
    if not isinstance(overrides, dict):
        raise TypeError("Solver settings must be a mapping")
    result = deepcopy(defaults)
    for key, value in overrides.items():
        if key not in defaults:
            raise KeyError(f"Unsupported solver key: {key}")
        result[key] = merge_settings(defaults[key], value) if isinstance(defaults[key], dict) else value
    return result


def load_config(dataset_dir, *, overrides=None):
    """config.yaml is required; config_neurosr.yaml and every solver key are optional."""
    directory = Path(dataset_dir).expanduser().resolve()
    facts = read_yaml(directory / "config.yaml")
    solver_path = directory / "config_neurosr.yaml"
    solver = merge_settings(DEFAULTS, read_yaml(solver_path) if solver_path.is_file() else {})
    if overrides is not None:
        solver = merge_settings(solver, overrides)
    for key in ("data", "sensor", "optics", "acquisition"):
        if not isinstance(facts.get(key), dict):
            raise KeyError(f"Missing measurement mapping: {key}")
    shape = facts["sensor"]["shape_yx"]
    if len(shape) != 2 or any(int(x) != x or x < 2 for x in shape):
        raise ValueError("sensor.shape_yx must contain two positive raster dimensions")
    for key in ("wavelength_m", "pixel_pitch_m"):
        value = facts["optics"][key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"optics.{key} must be a positive finite number")
    for key, minimum in (("pieces", 1), ("M", 2)):
        if not isinstance(solver[key], int) or isinstance(solver[key], bool) or solver[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in solver["iterations"].values()):
        raise ValueError("Each stage needs a positive integer iteration count")
    scan = solver["autofocus"]
    if scan["min_m"] <= 0 or scan["max_m"] <= scan["min_m"] or not isinstance(scan["samples"], int) or scan["samples"] < 2:
        raise ValueError("Autofocus needs 0 < min_m < max_m and at least two samples")
    if not isinstance(solver["seed"], int) or solver["sigma"] <= 0:
        raise ValueError("seed must be an integer and sigma must be positive")
    for stage, schedule in solver["schedule"].items():
        if len(schedule) != 2 or not isinstance(schedule[0], int) or schedule[0] < 1 or schedule[1] <= 0:
            raise ValueError(f"Invalid StepLR schedule for {stage}")
    # This route uses the frame alone at M = 1 and holds the transport and background after CMax.
    fixed = {"event": 0.0, "bg": 0.0, "tv": 0.0, "tv_sr": 0.0}
    if any(solver["kappa"][k] != v for k, v in fixed.items()) or solver["lr"]["motion"] != 0 or solver["lr"]["background"] != 0:
        raise ValueError("This route requires frame-only M=1, fixed motion/background, and zero sensor TV weights")
    return {"dataset_dir": directory, "measurement": facts, "solver": solver}
