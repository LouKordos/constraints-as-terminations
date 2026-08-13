from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = ROOT / "exts" / "locomposition"


def test_extension_distribution_and_package_are_named_locomposition():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extension_setup = (EXTENSION_ROOT / "setup.py").read_text()

    assert metadata["tool"]["uv"]["sources"]["locomposition"] == {
        "path": "exts/locomposition",
        "editable": True,
    }
    assert 'name="locomposition"' in extension_setup
    assert (EXTENSION_ROOT / "locomposition" / "__init__.py").is_file()
    assert 'include=("locomposition", "locomposition.*")' in extension_setup


def test_cat_algorithm_module_keeps_its_mechanism_name():
    assert (
        EXTENSION_ROOT
        / "locomposition"
        / "tasks"
        / "utils"
        / "cat"
        / "constraint_manager.py"
    ).is_file()
