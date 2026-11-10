"""Compare five-update CPU outputs with small reference arrays.

The fixture records seed, threads and sample indices.
"""
import json
from pathlib import Path
import numpy as np
import pytest

REFERENCE = Path(__file__).resolve().parent / "reference"
NAMES = ("object_m1", "object_sr", "latent_m1", "latent_sr", "trajectory", "c", "z", "z_m1", "z_autofocus", "background")


@pytest.mark.parametrize("name", NAMES)
def test_original_cpu_reference(cpu_result, name):
    arrays = cpu_result.arrays()
    actual = arrays[name]
    if actual.ndim == 2 and name != "trajectory":
        iy = np.linspace(0, actual.shape[0] - 1, 16).astype(int)
        ix = np.linspace(0, actual.shape[1] - 1, 16).astype(int)
        actual = actual[np.ix_(iy, ix)]
    with np.load(REFERENCE / "cpu_5iter.npz", allow_pickle=False) as stored:
        expected = stored[name]
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=5e-7)
    assert cpu_result.object_sr.device.type == "cpu"


def test_reference_provenance(cpu_result):
    provenance = json.loads((REFERENCE / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["iterations"] == cpu_result.config["solver"]["iterations"]
    assert provenance["seed"] == cpu_result.config["solver"]["seed"] == 0
    assert provenance["device"] == "cpu"
    assert provenance["source"].startswith("reference implementation")


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
