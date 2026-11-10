import json

import numpy as np
import pytest
import torch

from neurosr import joint_demo
from neurosr.joint import native_box
from neurosr.ops import down_sampling


def test_joint_keeps_aps_clock(tmp_path, monkeypatch):
    window = tmp_path / "window.npz"
    timestamps = np.array([2, 7, 11, 20, 51, 57])
    coordinates = np.full(timestamps.shape, 4, dtype=np.int16)
    np.savez(window, xs=coordinates, ys=coordinates, ts=timestamps,
             ps=np.ones_like(timestamps, dtype=np.int8),
             frames=np.full((2, 8, 10), 120, dtype=np.uint8),
             exposures_us=np.array([[0, 30], [50, 80]]),
             metadata=json.dumps({"scene": "earthworm", "frame_index": 7, "window_us": 100_000}))

    def check_motion_clock(solver, **kwargs):
        torch.testing.assert_close(solver.event.tt, torch.tensor(timestamps - 2, dtype=torch.float32))

    monkeypatch.setattr(joint_demo, "register_segments", lambda *args: torch.zeros((12, 2)))
    monkeypatch.setattr(joint_demo.NeuroSR, "cmax", check_motion_clock)
    _, events, _, _, _, _, metadata, scale = joint_demo.prepare(window, torch.device("cpu"))
    torch.testing.assert_close(events[2], torch.tensor(timestamps, dtype=torch.float32))
    assert metadata["first_event_offset_us"] == 2
    assert metadata["reference_us"] == 15
    assert scale == 3


@pytest.mark.parametrize("scale", [2, 3])
def test_native_box_matches_frame_pooling(scale):
    image = torch.rand((6 * scale, 7 * scale), generator=torch.Generator().manual_seed(0))
    offset = (scale - 1) // 2
    centre_samples = native_box(image, scale)[offset::scale, offset::scale]
    torch.testing.assert_close(centre_samples, down_sampling(image, scale))
