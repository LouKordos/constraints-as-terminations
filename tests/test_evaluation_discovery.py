from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_run_range import discover_metrics_summary_files, extract_checkpoint_from_path
from gait_dynamics_aggregate import discover_rebuttal_gait_dynamics_files


ENV_NAME = "env"
RUN_NAME = "2026-01-01-00-00-00"


def _make_summary(
    *,
    rebuttal_scenarios_only: bool,
    action_delay_steps: int | None,
    include_gait_dynamics: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "env_name": ENV_NAME,
        "run_name": RUN_NAME,
        "seed": 46,
        "rebuttal_scenarios_only": rebuttal_scenarios_only,
        "fixed_command_scenarios_metrics": {},
    }
    if action_delay_steps is not None:
        summary["action_delay_steps"] = action_delay_steps
    if include_gait_dynamics:
        summary["fixed_command_scenarios_gait_dynamics_metrics"] = {
            "walk_x_flat_terrain_1.0mps": {"gait_dynamics": {}}
        }
    return summary


def _write_summary(eval_dir: Path, summary: dict[str, Any]) -> Path:
    eval_dir.mkdir(parents=True)
    json_path = eval_dir / "metrics_summary.json"
    json_path.write_text(json.dumps(summary), encoding="utf-8")
    return json_path


def test_full_zero_delay_summary_is_selected_for_existing_gait_outputs(tmp_path: Path) -> None:
    json_path = _write_summary(
        tmp_path / "run" / "eval_checkpoint_100_seed_46_action_delay_0",
        _make_summary(rebuttal_scenarios_only=False, action_delay_steps=0),
    )

    selected, manifest = discover_rebuttal_gait_dynamics_files([tmp_path])

    entry = selected[(ENV_NAME, RUN_NAME)]
    assert entry.json_path == json_path
    assert entry.evaluation_scope == "full"
    selected_row = manifest.loc[manifest["selected"]].iloc[0]
    assert selected_row["selection_reason"] == "highest_checkpoint_full_zero_delay"


def test_rebuttal_only_is_fallback_and_delayed_eval_is_ineligible(tmp_path: Path) -> None:
    rebuttal_path = _write_summary(
        tmp_path / "run" / "eval_checkpoint_100_seed_46_action_delay_0_rebuttal_scenarios",
        _make_summary(rebuttal_scenarios_only=True, action_delay_steps=0),
    )
    _write_summary(
        tmp_path / "run" / "eval_checkpoint_101_seed_46_action_delay_1",
        _make_summary(rebuttal_scenarios_only=False, action_delay_steps=1),
    )

    selected, manifest = discover_rebuttal_gait_dynamics_files([tmp_path])

    entry = selected[(ENV_NAME, RUN_NAME)]
    assert entry.json_path == rebuttal_path
    assert entry.evaluation_scope == "rebuttal_only"
    delayed = manifest.loc[manifest["action_delay_steps"] == 1].iloc[0]
    assert not bool(delayed["eligible"])
    assert delayed["selection_reason"] == "excluded_nonzero_action_delay"


def test_full_scope_is_preferred_before_checkpoint_for_gait_outputs(tmp_path: Path) -> None:
    full_path = _write_summary(
        tmp_path / "run" / "eval_checkpoint_100_seed_46_action_delay_0",
        _make_summary(rebuttal_scenarios_only=False, action_delay_steps=0),
    )
    _write_summary(
        tmp_path / "run" / "eval_checkpoint_101_seed_46_action_delay_0_rebuttal_scenarios",
        _make_summary(rebuttal_scenarios_only=True, action_delay_steps=0),
    )

    selected, _ = discover_rebuttal_gait_dynamics_files([tmp_path])

    assert selected[(ENV_NAME, RUN_NAME)].json_path == full_path


def test_general_discovery_accepts_suffixed_paths_and_prefers_current_full_data(
    tmp_path: Path,
) -> None:
    old_path = _write_summary(
        tmp_path / "run" / "eval_checkpoint_100",
        _make_summary(
            rebuttal_scenarios_only=False,
            action_delay_steps=None,
            include_gait_dynamics=False,
        ),
    )
    current_path = _write_summary(
        tmp_path / "run" / "eval_checkpoint_100_seed_46_action_delay_0",
        _make_summary(rebuttal_scenarios_only=False, action_delay_steps=0),
    )

    assert extract_checkpoint_from_path(current_path) == 100
    selected, manifest = discover_metrics_summary_files(
        root_dir=tmp_path,
        cot_scenario_pattern=r"^cot_(\d+(?:\.\d+)?)$",
        cot_velocity_range=(0.6, 1.6),
    )

    assert selected[(ENV_NAME, RUN_NAME)].json_path == current_path.resolve()
    selected_row = manifest.loc[manifest["selected"]].iloc[0]
    assert selected_row["json_path"] == str(current_path.resolve())
    assert selected_row["selection_reason"] == "current_schema_tiebreak"
    assert not bool(manifest.loc[manifest["json_path"] == str(old_path.resolve()), "selected"].item())

