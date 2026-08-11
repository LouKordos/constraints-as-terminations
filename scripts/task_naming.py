"""Naming helpers for canonical and compatibility Gymnasium task IDs."""

from __future__ import annotations


LEGACY_PREFIX = "CaT-"
CANONICAL_PREFIX = "LoComposition-"
PROJECT_FAMILIES = ("Go2-", "Anymal-C-", "Spot-")


def canonical_task_id(task_id: str) -> str:
    """Return the LoComposition name for a legacy project task ID."""

    if any(
        task_id.startswith(LEGACY_PREFIX + family) for family in PROJECT_FAMILIES
    ):
        return CANONICAL_PREFIX + task_id.removeprefix(LEGACY_PREFIX)
    return task_id


def is_locomposition_task(task_id: str) -> bool:
    """Return whether an ID belongs to a current or legacy project task."""

    canonical_id = canonical_task_id(task_id)
    return any(
        canonical_id.startswith(CANONICAL_PREFIX + family)
        for family in PROJECT_FAMILIES
    )


def is_locomposition_go2_task(task_id: str) -> bool:
    """Return whether an ID belongs to the current or legacy Go2 task family."""

    return canonical_task_id(task_id).startswith(CANONICAL_PREFIX + "Go2-")
