"""Rewrite tests/reference from the current code.

    python tests/make_reference.py

Run it only after an intended numerical change, and review the difference it makes to the stored
arrays.  It writes the samples that the tests compare and, in provenance.json, the software
versions and the SHA-256 of every full array.
"""
import json
import platform
import tempfile
from pathlib import Path

import numpy as np
import torch

import reference_runs as runs


def main():
    stored, hashes = {}, {}
    for name, overrides in (("cpu_5iter", runs.FIVE), ("cpu_5iter_stepped", runs.STEPPED)):
        arrays = runs.holography(overrides).arrays()
        stored[name] = {key: runs.sample(value) for key, value in arrays.items()}
        hashes[name] = {key: runs.digest(value) for key, value in arrays.items()}
    with tempfile.TemporaryDirectory() as scratch:
        for scene in runs.SCENES:
            arrays = runs.joint(scene, Path(scratch) / scene)
            stored[f"joint_tiny_{scene}"] = {key: runs.sample(value) for key, value in arrays.items()}
            hashes[f"joint_tiny_{scene}"] = {key: runs.digest(value) for key, value in arrays.items()}
        with runs.stepped_joint_schedules():
            arrays = runs.joint("earthworm", Path(scratch) / "stepped")
        stored["joint_tiny_stepped_earthworm"] = {key: runs.sample(value) for key, value in arrays.items()}
        hashes["joint_tiny_stepped_earthworm"] = {key: runs.digest(value) for key, value in arrays.items()}
    for name, arrays in stored.items():
        np.savez_compressed(runs.REFERENCE / f"{name}.npz", **arrays)
    provenance = dict(
        device="cpu", threads=1, seed=0, updates_per_stage=5,
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        platform=platform.platform(), machine=platform.machine(),
        sample="images: 16 x 16 evenly spaced pixels including the corners; stacks over the "
               "transport segments: summed over the segments first; trajectories and scalars: complete",
        stepped_schedule=runs.STEPPED["schedule"], stepped_joint_schedule=[2, 0.5], full_array_hashes=hashes)
    (runs.REFERENCE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                                    encoding="utf-8", newline="\n")
    print("Wrote", ", ".join(sorted(stored)), "to", runs.REFERENCE)


if __name__ == "__main__":
    main()
