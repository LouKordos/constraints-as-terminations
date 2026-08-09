from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gait_dynamics_aggregate
from gait_dynamics_aggregate import (
    _render_aggregate_report,
    aggregate_rebuttal_gait_dynamics,
)


FIVE_SCENARIOS = (
    "walk_x_flat_terrain_1.0mps",
    "fast_walk_stairs_up",
    "medium_walk_diagonal_turning_uneven_terrain",
    "fast_walk_diagonal_uneven_terrain",
    "fast_walk_x_uneven_terrain",
)


def _per_run_dataframe(*, second_run_missing_last: bool = False) -> pd.DataFrame:
    rows = []
    for run_name in ("run-a", "run-b"):
        scenarios = FIVE_SCENARIOS
        if second_run_missing_last and run_name == "run-b":
            scenarios = FIVE_SCENARIOS[:-1]
        for scenario_order, scenario in enumerate(scenarios):
            rows.append(
                {
                    "label": "LEP",
                    "env_name": "env",
                    "run_name": run_name,
                    "checkpoint": 100,
                    "eval_seed": 46,
                    "action_delay_steps": 0,
                    "scenario": scenario,
                    "scenario_order": scenario_order,
                    "completion_percent": 100.0,
                    "aerial_phase_percent": 10.0,
                    "vertical_acceleration_rms_g": 1.0,
                    "total_vertical_grf_p95_body_weight": 2.0,
                    "joint_acceleration_rms_rad_s2": 100.0,
                    "base_acceleration_fidelity": "direct",
                    "vertical_grf_fidelity": "direct_vector_history",
                }
            )
    return pd.DataFrame(rows)


def test_report_uses_dynamic_configured_scenario_count(monkeypatch) -> None:
    monkeypatch.setattr(
        gait_dynamics_aggregate,
        "REBUTTAL_DYNAMICS_SCENARIOS",
        FIVE_SCENARIOS,
    )
    per_run = _per_run_dataframe()
    per_scenario, _, overall = aggregate_rebuttal_gait_dynamics(per_run)

    report = _render_aggregate_report(per_run, per_scenario, overall)

    assert "5 configured scenarios" in report
    assert "four scenarios" not in report.lower()


def test_report_names_run_with_missing_configured_scenario(monkeypatch) -> None:
    monkeypatch.setattr(
        gait_dynamics_aggregate,
        "REBUTTAL_DYNAMICS_SCENARIOS",
        FIVE_SCENARIOS,
    )
    per_run = _per_run_dataframe(second_run_missing_last=True)
    per_scenario, _, overall = aggregate_rebuttal_gait_dynamics(per_run)

    report = _render_aggregate_report(per_run, per_scenario, overall)

    assert "LEP/run-b" in report
    assert "fast_walk_x_uneven_terrain" in report
