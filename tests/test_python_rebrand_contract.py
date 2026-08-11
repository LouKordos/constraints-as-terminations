from __future__ import annotations

import importlib
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = ROOT / "exts" / "locomposition"


def test_primary_distribution_and_extension_are_named_locomposition():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert metadata["project"]["name"] == "locomposition"
    assert (EXTENSION_ROOT / "locomposition" / "__init__.py").is_file()


@pytest.mark.parametrize("legacy_first", [False, True])
def test_legacy_namespace_resolves_to_canonical_module_objects(
    monkeypatch, tmp_path, legacy_first
):
    canonical = tmp_path / "sample_project"
    child = canonical / "child"
    child.mkdir(parents=True)
    (canonical / "__init__.py").write_text("VALUE = object()\n")
    (child / "__init__.py").write_text("from .. import VALUE\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    helper_path = EXTENSION_ROOT / "locomposition" / "_legacy_namespace.py"
    spec = importlib.util.spec_from_file_location("legacy_namespace_test", helper_path)
    assert spec is not None and spec.loader is not None
    alias_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alias_module)
    original_meta_path = list(sys.meta_path)
    try:
        alias_module.install_legacy_namespace(
            canonical_root="sample_project", legacy_root="sample_legacy"
        )

        roots = ("sample_legacy", "sample_project") if legacy_first else (
            "sample_project",
            "sample_legacy",
        )
        children = (
            ("sample_legacy.child", "sample_project.child")
            if legacy_first
            else ("sample_project.child", "sample_legacy.child")
        )

        imported_roots = {name: importlib.import_module(name) for name in roots}
        imported_children = {
            name: importlib.import_module(name) for name in children
        }
        canonical_root = imported_roots["sample_project"]
        canonical_child = imported_children["sample_project.child"]
        legacy_root = imported_roots["sample_legacy"]
        legacy_child = imported_children["sample_legacy.child"]

        assert legacy_root is canonical_root
        assert legacy_child is canonical_child
        assert legacy_child.VALUE is canonical_root.VALUE
    finally:
        sys.meta_path[:] = original_meta_path
        for name in (
            "sample_legacy.child",
            "sample_legacy",
            "sample_project.child",
            "sample_project",
        ):
            sys.modules.pop(name, None)


def test_cat_algorithm_module_keeps_its_mechanism_name():
    assert (
        EXTENSION_ROOT
        / "locomposition"
        / "tasks"
        / "utils"
        / "cat"
        / "constraint_manager.py"
    ).is_file()
