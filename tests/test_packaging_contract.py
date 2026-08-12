from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def load_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def project_requirements() -> dict[str, Requirement]:
    project = load_toml(ROOT / "pyproject.toml")
    return {
        requirement.name.lower(): requirement
        for raw_requirement in project["project"]["dependencies"]
        if (requirement := Requirement(raw_requirement))
    }


def test_root_is_a_non_package_uv_environment_project():
    project = load_toml(ROOT / "pyproject.toml")

    assert project["project"]["name"] == "locomposition-environment"
    assert project["project"]["requires-python"] == "==3.11.*"
    assert project["tool"]["uv"]["package"] is False
    assert "build-system" not in project


def test_project_declares_all_previous_runtime_dependencies():
    requirements = project_requirements()
    expected_names = {
        "isaacsim",
        "locomposition",
        "matplotlib",
        "memory-profiler",
        "numpy",
        "pandas",
        "pyqt5",
        "rich",
        "rosbags",
        "rsl-rl-lib",
        "scienceplots",
        "scikit-learn",
        "scipy",
        "seaborn",
        "statsmodels",
        "starlette",
        "tensorboard",
        "tensordict",
        "toml",
        "torch",
        "torchvision",
        "tqdm",
        "wandb",
        "zarr",
        "zmq",
    }

    assert expected_names <= requirements.keys()
    assert str(requirements["torch"].specifier) == "==2.7.0"
    assert str(requirements["torchvision"].specifier) == "==0.22.0"
    assert str(requirements["isaacsim"].specifier) == "==5.1.0"
    assert requirements["isaacsim"].extras == {"all", "extscache"}
    assert str(requirements["numpy"].specifier) == "==1.26.0"
    assert str(requirements["zarr"].specifier) == "==3.1.5"
    assert str(requirements["rsl-rl-lib"].specifier) == "==5.4.2"
    assert str(requirements["starlette"].specifier) == "==0.49.1"


def test_uv_sources_keep_indexes_and_install_local_extension_editably():
    project = load_toml(ROOT / "pyproject.toml")
    uv = project["tool"]["uv"]
    indexes = {index["name"]: index for index in uv["index"]}
    sources = uv["sources"]

    assert indexes["pytorch"] == {
        "name": "pytorch",
        "url": "https://download.pytorch.org/whl/cu128",
        "explicit": True,
    }
    assert indexes["nvidia"] == {
        "name": "nvidia",
        "url": "https://pypi.nvidia.com",
        "explicit": True,
    }
    assert sources["torch"] == {"index": "pytorch"}
    assert sources["torchvision"] == {"index": "pytorch"}
    assert sources["isaacsim"] == {"index": "nvidia"}
    assert sources["locomposition"] == {
        "path": "exts/locomposition",
        "editable": True,
    }
    assert uv["override-dependencies"] == [
        "pywin32; sys_platform == 'win32'",
        "starlette==0.49.1",
    ]


def test_extension_declares_an_isolated_setuptools_build():
    extension = load_toml(ROOT / "exts" / "locomposition" / "pyproject.toml")

    assert extension["build-system"] == {
        "requires": ["setuptools", "wheel", "toml"],
        "build-backend": "setuptools.build_meta",
    }


def test_legacy_requirements_file_is_removed_and_lock_is_committed():
    assert not (ROOT / "requirements.txt").exists()
    assert (ROOT / "uv.lock").is_file()
