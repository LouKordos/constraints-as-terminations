"""Import aliases for compatibility with the former project package name."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType


class _LegacyAliasLoader(importlib.abc.Loader):
    """Return an already-imported canonical module for a legacy module name."""

    def __init__(self, canonical_name: str):
        self.canonical_name = canonical_name

    def create_module(self, spec) -> ModuleType:
        return importlib.import_module(self.canonical_name)

    def exec_module(self, module: ModuleType) -> None:
        return None


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    """Map descendants of one import root onto another import root."""

    def __init__(self, canonical_root: str, legacy_root: str):
        self.canonical_root = canonical_root
        self.legacy_root = legacy_root

    def find_spec(self, fullname: str, path=None, target=None):
        legacy_prefix = f"{self.legacy_root}."
        if not fullname.startswith(legacy_prefix):
            return None

        suffix = fullname.removeprefix(legacy_prefix)
        canonical_name = f"{self.canonical_root}.{suffix}"
        canonical_spec = importlib.util.find_spec(canonical_name)
        if canonical_spec is None:
            return None

        return importlib.util.spec_from_loader(
            fullname,
            _LegacyAliasLoader(canonical_name),
            is_package=canonical_spec.submodule_search_locations is not None,
        )


def install_legacy_namespace(*, canonical_root: str, legacy_root: str) -> None:
    """Install ``legacy_root`` as an object-identical alias of ``canonical_root``."""

    if not any(
        isinstance(finder, _LegacyAliasFinder)
        and finder.canonical_root == canonical_root
        and finder.legacy_root == legacy_root
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _LegacyAliasFinder(canonical_root, legacy_root))

    sys.modules[legacy_root] = importlib.import_module(canonical_root)
