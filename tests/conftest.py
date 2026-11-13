"""Seeded single-thread CPU reconstructions shared by the reference checks."""
import pytest

import reference_runs


def pytest_addoption(parser):
    parser.addoption("--bitwise", action="store_true",
                     help="bit-level comparison with the stored references (see tests/reference_runs.py)")


@pytest.fixture(scope="session")
def bitwise(request):
    return request.config.getoption("--bitwise")


@pytest.fixture(scope="session")
def cpu_result():
    return reference_runs.holography(reference_runs.FIVE)


@pytest.fixture(scope="session")
def stepped_result():
    return reference_runs.holography(reference_runs.STEPPED)


@pytest.fixture(scope="session", params=reference_runs.SCENES)
def joint_result(request, tmp_path_factory):
    scene = request.param
    return scene, reference_runs.joint(scene, tmp_path_factory.mktemp("joint") / scene)


@pytest.fixture(scope="session")
def stepped_joint_result(tmp_path_factory):
    with reference_runs.stepped_joint_schedules():
        return reference_runs.joint("earthworm", tmp_path_factory.mktemp("stepped") / "earthworm")
