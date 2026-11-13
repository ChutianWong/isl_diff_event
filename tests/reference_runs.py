"""Short deterministic CPU runs of the three bundled windows, and how they are compared.

Every run uses one thread and five updates per stage.  An image is compared on 16 x 16 evenly
spaced pixels, a stack over the transport segments after summing over the segments, a trajectory
or a scalar in full.  By default floating-point values must agree with the stored reference to
1e-3 in relative L2 norm for images and 1e-5 for trajectories and scalars, which holds across CPU
instruction sets; two of them differ by up to 1.5e-4 after five updates.  Integer values must be
equal.  With --bitwise the samples must agree to 5e-7 and the SHA-256 of every full array must
match, which holds on the platform that wrote the reference (provenance.json).
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from neurosr import NeuroSR, joint as joint_fits, joint_demo

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/holography/usaf_veryclose"
REFERENCE = Path(__file__).resolve().parent / "reference"
FIVE = {"iterations": {"cmax": 5, "joint": 5, "sr": 5}}
# The same five updates with every learning-rate step inside them.
STEPPED = dict(FIVE, schedule={"cmax": [2, 0.5], "joint": [2, 0.5], "sr": [2, 0.5]})
SCENES = ("earthworm", "testis")


@contextmanager
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@contextmanager
def stepped_joint_schedules():
    """The intensity fits of the joint route with every learning-rate step inside five updates."""
    saved = joint_fits.NATIVE["schedule"], joint_fits.FINE["schedule"]
    joint_fits.NATIVE["schedule"] = joint_fits.FINE["schedule"] = (2, 0.5)
    try:
        yield
    finally:
        joint_fits.NATIVE["schedule"], joint_fits.FINE["schedule"] = saved


def holography(overrides):
    """The solver of the bundled USAF window after its three stages."""
    with one_thread():
        return NeuroSR.from_directory(DATASET, device="cpu", overrides=overrides).run()


def joint(scene, out):
    """The arrays of `python -m neurosr joint data/joint/<scene>.npz --tiny`, with the fitted
    contrast threshold and, for a complex field, the propagation distance."""
    with one_thread():
        joint_demo.main([str(ROOT / "data/joint" / f"{scene}.npz"), "--output", str(out),
                         "--device", "cpu", "--threads", "1", "--tiny"])
    with np.load(out / "result.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    report = json.loads((out / "result.json").read_text(encoding="utf-8"))
    arrays["contrast"] = np.asarray(report["contrast"])
    if "z_m" in report["fits"]["sr"]:
        arrays["z"] = np.asarray(report["fits"]["sr"]["z_m"])
    return arrays


def sample(array):
    array = np.asarray(array)
    if array.ndim == 3:
        array = array.sum(0)
    if array.dtype.kind in "biu":
        array = array.astype(np.int64)
    if array.ndim == 2 and min(array.shape) >= 16:
        rows = np.linspace(0, array.shape[0] - 1, 16).astype(int)
        columns = np.linspace(0, array.shape[1] - 1, 16).astype(int)
        array = array[np.ix_(rows, columns)]
    return array


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def assert_matches(actual, expected, bitwise):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if actual.dtype.kind in "biu":
        np.testing.assert_array_equal(actual, expected)
    elif bitwise:
        np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=5e-7)
    else:
        tolerance = 1e-5 if actual.size <= 64 else 1e-3
        assert np.linalg.norm(actual - expected) <= tolerance * np.linalg.norm(expected) + 1e-12
