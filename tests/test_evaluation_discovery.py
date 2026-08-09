from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_run_range import (
    SeriesData,
    build_terrain_run_sources,
    discover_metrics_summary_files,
    extract_checkpoint_from_path,
)
from gait_dynamics_aggregate import discover_rebuttal_gait_dynamics_files


ENV_NAME = "env"
RUN_NAME = "2026-01-01-00-00-00"


def _make_summary(
    *,
    rebuttal_scenarios_only: bool,
    action_delay_steps: int | None,
    include_gait_dynamics: bool = True,
    scenario_count: int = 1,
    selected_fixed_scenario: str | None = None,
) -> dict[str, Any]:
    scenario_tags = [f"scenario_{index}" for index in range(scenario_count)]
    summary: dict[str, Any] = {
        "env_name": ENV_NAME,
        "run_name": RUN_NAME,
        "seed": 46,
        "rebuttal_scenarios_only": rebuttal_scenarios_only,
        "selected_fixed_scenario": selected_fixed_scenario,
        "fixed_command_scenarios": [
            [tag, [1.0, 0.0, 0.0], None] for tag in scenario_tags
        ],
        "fixed_command_scenarios_metrics": {tag: {} for tag in scenario_tags},
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
    assert selected_row["selection_reason"] == "checkpoint_scenario_count_delay_path_rank"


def test_higher_checkpoint_wins_even_with_fewer_scenarios(tmp_path: Path) -> None:
    _write_summary(
        tmp_path / "run" / "eval_checkpoint_100_seed_46_action_delay_0_rebuttal_scenarios",
        _make_summary(
            rebuttal_scenarios_only=False,
            action_delay_steps=0,
            scenario_count=21,
        ),
    )
    newer_path = _write_summary(
        tmp_path / "run" / "eval_checkpoint_101_seed_46_action_delay_1",
        _make_summary(
            rebuttal_scenarios_only=False,
            action_delay_steps=1,
            scenario_count=1,
            selected_fixed_scenario="scenario_0",
        ),
    )

    selected, manifest = discover_rebuttal_gait_dynamics_files([tmp_path])

    entry = selected[(ENV_NAME, RUN_NAME)]
    assert entry.json_path == newer_path
    assert entry.evaluation_scope == "single_scenario"
    selected_row = manifest.loc[manifest["selected"]].iloc[0]
    assert selected_row["checkpoint"] == 101


def test_more_scenarios_win_at_the_same_checkpoint(tmp_path: Path) -> None:
    _write_summary(
        tmp_path / "run" / "eval_checkpoint_100_seed_46_action_delay_0_rebuttal_scenarios",
        _make_summary(
            rebuttal_scenarios_only=True,
            action_delay_steps=0,
            scenario_count=4,
        ),
    )
    full_path = _write_summary(
        tmp_path / "run" / "eval_checkpoint_100_seed_46_action_delay_1",
        _make_summary(
            rebuttal_scenarios_only=False,
            action_delay_steps=1,
            scenario_count=21,
        ),
    )

    selected, _ = discover_rebuttal_gait_dynamics_files([tmp_path])

    assert selected[(ENV_NAME, RUN_NAME)].json_path == full_path


def test_zero_delay_wins_after_checkpoint_and_scenario_count(tmp_path: Path) -> None:
    _write_summary(
        tmp_path / "run" / "eval_checkpoint_100_seed_46_action_delay_2",
        _make_summary(
            rebuttal_scenarios_only=False,
            action_delay_steps=2,
            scenario_count=21,
        ),
    )
    zero_delay_path = _write_summary(
        tmp_path / "run" / "eval_checkpoint_100_seed_46_action_delay_0",
        _make_summary(
            rebuttal_scenarios_only=False,
            action_delay_steps=0,
            scenario_count=21,
        ),
    )

    selected, _ = discover_rebuttal_gait_dynamics_files([tmp_path])

    assert selected[(ENV_NAME, RUN_NAME)].json_path == zero_delay_path


def test_general_discovery_accepts_suffixed_paths_and_uses_deterministic_path_tiebreak(
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
        _make_summary(
            rebuttal_scenarios_only=False,
            action_delay_steps=0,
            include_gait_dynamics=False,
        ),
    )

    assert extract_checkpoint_from_path(current_path) == 100
    selected, manifest = discover_metrics_summary_files(
        root_dir=tmp_path,
        cot_scenario_pattern=r"^cot_(\d+(?:\.\d+)?)$",
        cot_velocity_range=(0.6, 1.6),
    )

    assert selected[(ENV_NAME, RUN_NAME)].json_path == old_path.resolve()
    selected_row = manifest.loc[manifest["selected"]].iloc[0]
    assert selected_row["json_path"] == str(old_path.resolve())
    assert selected_row["selection_reason"] == "checkpoint_scenario_count_delay_path_rank"
    assert not bool(
        manifest.loc[
            manifest["json_path"] == str(current_path.resolve()), "selected"
        ].item()
    )


def test_selected_full_non_rebuttal_eval_becomes_a_terrain_analysis_source(
    tmp_path: Path,
) -> None:
    _write_summary(
        tmp_path / "run" / "eval_checkpoint_100_seed_46_action_delay_0",
        _make_summary(rebuttal_scenarios_only=False, action_delay_steps=0),
    )
    selected, _ = discover_metrics_summary_files(
        root_dir=tmp_path,
        cot_scenario_pattern=r"^cot_(\d+(?:\.\d+)?)$",
        cot_velocity_range=(0.6, 1.6),
    )
    json_run = selected[(ENV_NAME, RUN_NAME)]
    series_data = {
        "LEP": SeriesData(
            label="LEP",
            env_name=ENV_NAME,
            wandb_path=ENV_NAME,
            wandb_runs={},
            json_runs={RUN_NAME: json_run},
            selected_wandb_run_names=[],
            selected_json_run_names=[RUN_NAME],
        )
    }

    sources = build_terrain_run_sources(series_data)

    assert len(sources) == 1
    assert sources[0].label == "LEP"
    assert sources[0].run_name == RUN_NAME
    assert sources[0].json_path == json_run.json_path
