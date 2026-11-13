"""Short CPU reconstructions of the bundled windows against stored references.

tests/reference_runs.py states what is compared and to which tolerance.
"""
import json

import numpy as np
import pytest

from reference_runs import REFERENCE, assert_matches, digest, sample

PROVENANCE = json.loads((REFERENCE / "provenance.json").read_text(encoding="utf-8"))


def check(name, arrays, bitwise):
    with np.load(REFERENCE / f"{name}.npz", allow_pickle=False) as stored:
        assert set(stored.files) == set(arrays)
        for key in stored.files:
            assert_matches(sample(arrays[key]), stored[key], bitwise)
    if bitwise:
        assert {key: digest(value) for key, value in arrays.items()} == PROVENANCE["full_array_hashes"][name]


def test_holography_reference(cpu_result, bitwise):
    check("cpu_5iter", cpu_result.arrays(), bitwise)
    assert cpu_result.object_sr.device.type == "cpu"


def test_holography_reference_with_learning_rate_steps(cpu_result, stepped_result, bitwise):
    check("cpu_5iter_stepped", stepped_result.arrays(), bitwise)
    plain, stepped = cpu_result.arrays(), stepped_result.arrays()
    for key in ("trajectory", "object_m1", "object_sr"):
        assert np.abs(plain[key] - stepped[key]).max() > 1e-4


def test_joint_reference(joint_result, bitwise):
    scene, arrays = joint_result
    check(f"joint_tiny_{scene}", arrays, bitwise)
    scale = 3 if scene == "earthworm" else 2
    assert arrays["joint"].shape == (260 * scale, 346 * scale)
    assert arrays["iwe_measured"].shape == arrays["iwe_predicted"].shape == (12, 260 * scale, 346 * scale)
    assert ("object_sr" in arrays) == (scene == "testis")


def test_joint_reference_with_learning_rate_steps(joint_result, stepped_joint_result, bitwise):
    check("joint_tiny_stepped_earthworm", stepped_joint_result, bitwise)
    scene, plain = joint_result
    if scene == "earthworm":
        assert np.abs(plain["joint"] - stepped_joint_result["joint"]).max() > 1e-4


def test_save_contract_and_data_protection(cpu_result, tmp_path):
    out = cpu_result.save(tmp_path / "result")
    assert {"reconstruction.npz", "metadata.json", "summary.png", "iwe_native_and_sr.png"} <= {p.name for p in out.iterdir()}
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["solver"]["iterations"] == {"cmax": 5, "joint": 5, "sr": 5}
    assert metadata["environment"]["device"] == "cpu"
    assert metadata["dataset"] == cpu_result.config["dataset_dir"].name
    assert set(metadata["files"]) == {"raw/window.npz", "config.yaml", "config_neurosr.yaml"}
    with np.load(out / "reconstruction.npz", allow_pickle=False) as saved:
        assert saved["object_sr"].tobytes() == cpu_result.arrays()["object_sr"].tobytes()
    with pytest.raises(ValueError, match="outside data"):
        cpu_result.save(cpu_result.config["dataset_dir"] / "output")
