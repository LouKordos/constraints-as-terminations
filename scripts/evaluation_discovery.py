from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar


@dataclass(frozen=True)
class EvaluationMetadata:
    checkpoint: int | None
    action_delay_steps: int
    evaluation_scope: str
    fixed_scenario_count: int


def extract_checkpoint_from_path(path: Path) -> int | None:
    pattern = re.compile(r"^eval_checkpoint_(\d+)(?:_seed_\d+)?(?:_.*)?$")
    for part in reversed(path.parts):
        match = pattern.match(part)
        if match:
            return int(match.group(1))
    return None


def _infer_action_delay_from_path(path: Path) -> int:
    match = re.search(r"_action_delay_(\d+)(?:_|$)", str(path))
    return int(match.group(1)) if match else 0


def _fixed_scenario_count(summary: Mapping[str, Any]) -> int:
    scenarios = summary.get("fixed_command_scenarios")
    if isinstance(scenarios, list):
        return len(scenarios)
    scenario_metrics = summary.get("fixed_command_scenarios_metrics")
    if isinstance(scenario_metrics, dict):
        return len(scenario_metrics)
    return 0


def extract_evaluation_metadata(
    summary: Mapping[str, Any],
    json_path: Path,
) -> EvaluationMetadata:
    raw_action_delay = summary.get("action_delay_steps")
    action_delay_steps = (
        _infer_action_delay_from_path(json_path)
        if raw_action_delay is None
        else int(raw_action_delay)
    )
    if summary.get("selected_fixed_scenario"):
        evaluation_scope = "single_scenario"
    elif summary.get("rebuttal_scenarios_only") is True:
        evaluation_scope = "rebuttal_only"
    else:
        evaluation_scope = "full"
    return EvaluationMetadata(
        checkpoint=extract_checkpoint_from_path(json_path),
        action_delay_steps=action_delay_steps,
        evaluation_scope=evaluation_scope,
        fixed_scenario_count=_fixed_scenario_count(summary),
    )


def evaluation_rank_key(entry: Any) -> tuple[int, int, int, str]:
    checkpoint = entry.checkpoint if entry.checkpoint is not None else -1
    return (
        -int(checkpoint),
        -int(entry.fixed_scenario_count),
        0 if int(entry.action_delay_steps) == 0 else 1,
        str(entry.json_path),
    )


T = TypeVar("T")


def rank_evaluation_entries(entries: Iterable[T]) -> list[T]:
    return sorted(entries, key=evaluation_rank_key)


def select_preferred_evaluation(entries: Iterable[T]) -> T | None:
    ranked = rank_evaluation_entries(entries)
    return ranked[0] if ranked else None
