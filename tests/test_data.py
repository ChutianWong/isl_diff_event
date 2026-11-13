"""The bundled window retains recording values, dtypes and acquisition metadata."""
from pathlib import Path
import hashlib
import numpy as np
import pytest
from neurosr.config import DEFAULTS, load_config, merge_settings
from neurosr.data import load_window_npz

DATASET = Path(__file__).resolve().parents[1] / "data/holography/usaf_veryclose"


@pytest.mark.parametrize("key,dtype,shape", [
    ("xs", "int16", (239118,)), ("ys", "int16", (239118,)),
    ("ts", "int64", (239118,)), ("ps", "int8", (239118,)),
    ("frame", "uint8", (260, 346)),
])
def test_window_arrays_match_config(key, dtype, shape):
    config = load_config(DATASET)
    recording = load_window_npz(config)
    actual = getattr(recording, key)
    assert actual.dtype == np.dtype(dtype)
    assert actual.shape == shape
    with np.load(recording.source_path, allow_pickle=False) as stored:
        assert actual.tobytes() == stored[key].tobytes()


def test_measurement_facts_and_file_hash():
    config = load_config(DATASET)
    recording = load_window_npz(config)
    facts = config["measurement"]
    assert recording.frame_ind == facts["acquisition"]["frame_index"] == 82
    assert recording.win_frame == facts["acquisition"]["exposure_us"] == 2999
    assert recording.win == facts["acquisition"]["window_us"] == 100000
    assert recording.start == 3276296
    assert recording.end == 3279295
    assert facts["optics"]["wavelength_m"] == 632.8 * 1e-9
    assert facts["optics"]["pixel_pitch_m"] == 18.5 * 1e-6
    assert hashlib.sha256(recording.source_path.read_bytes()).hexdigest() == facts["data"]["sha256"]


def test_missing_solver_file_and_nested_keys_use_defaults(tmp_path):
    (tmp_path / "config.yaml").write_bytes((DATASET / "config.yaml").read_bytes())
    assert load_config(tmp_path)["solver"] == DEFAULTS
    (tmp_path / "config_neurosr.yaml").write_text("iterations:\n  joint: 5\n", encoding="utf-8")
    settings = load_config(tmp_path)["solver"]
    assert settings["iterations"] == {"cmax": 2100, "joint": 5, "sr": 2001}
    assert settings["lr"] == DEFAULTS["lr"]


def test_unknown_solver_keys_are_rejected():
    with pytest.raises(KeyError, match="Unsupported solver key"):
        merge_settings(DEFAULTS, {"kappa": {"pnp": 0.6}})


def test_window_that_disagrees_with_config_is_named(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/window.npz").write_bytes((DATASET / "raw/window.npz").read_bytes())
    facts = (DATASET / "config.yaml").read_text(encoding="utf-8").replace("frame_index: 82", "frame_index: 83")
    (tmp_path / "config.yaml").write_text(facts, encoding="utf-8")
    with pytest.raises(ValueError, match="frame_ind = 82 against 83"):
        load_window_npz(load_config(tmp_path))


JOINT_WINDOW = DATASET.parents[1] / "joint/earthworm.npz"


@pytest.mark.parametrize("arguments", [
    [str(DATASET), "--output", str(DATASET / "output")],
    [str(DATASET), "--output", str(DATASET.parents[1] / "output")],
    ["joint", str(JOINT_WINDOW), "--output", str(JOINT_WINDOW.parent / "output")],
])
def test_command_line_refuses_output_inside_data(arguments, monkeypatch, capsys):
    from neurosr import __main__ as cli
    monkeypatch.setattr("sys.argv", ["neurosr", *arguments, "--tiny", "--device", "cpu"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "outside data/" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [[str(DATASET)], ["joint", str(JOINT_WINDOW)]])
def test_command_line_refuses_an_existing_output(arguments, tmp_path, monkeypatch, capsys):
    from neurosr import __main__ as cli
    monkeypatch.setattr("sys.argv", ["neurosr", *arguments, "--output", str(tmp_path), "--tiny", "--device", "cpu"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "choose a new output directory" in capsys.readouterr().err


def test_joint_presets_have_a_recording():
    """Every solver preset names its recording in the tool that cuts its window."""
    import ast
    from neurosr.joint_demo import PRESETS
    source = (DATASET.parents[2] / "tools/make_joint_window.py").read_text(encoding="utf-8")
    assignment = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.Assign) and node.targets[0].id == "RECORDINGS")
    assert set(ast.literal_eval(assignment.value)) == set(PRESETS)
