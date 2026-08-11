"""Compatibility namespace for the former :mod:`cat_envs` package name."""

from locomposition._legacy_namespace import install_legacy_namespace


install_legacy_namespace(canonical_root="locomposition", legacy_root="cat_envs")
