"""One seeded CPU reconstruction shared by the equivalence checks."""
from pathlib import Path
import pytest
import torch
from neurosr import NeuroSR

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/holography/usaf_veryclose"


@pytest.fixture(scope="session")
def cpu_result():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    model = NeuroSR.from_directory(DATASET, device="cpu",
                                   overrides={"iterations": {"cmax": 5, "joint": 5, "sr": 5}})
    yield model.run()
    torch.set_num_threads(previous)
