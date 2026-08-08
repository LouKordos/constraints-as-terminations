from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from terrain_adaptation_analysis import (
    TerrainRunSource,
    aggregate_terrain_adaptation,
    analyze_terrain_sources,
    compute_scenario_contact_metrics,
    select_representative_run,
    write_terrain_adaptation_outputs,
)


FOOT_LABELS = ["RR_foot", "FL_foot", "FR_foot", "RL_foot"]


def _diagonal_contacts(phase_lengths: list[int]) -> np.ndarray:
    rows: list[list[bool]] = []
    for phase_index, phase_length in enumerate(phase_lengths):
        if phase_index % 2 == 0:
            row = [True, True, False, False]
        else:
            row = [False, False, True, True]
        rows.extend([row] * phase_length)
    return np.asarray(rows, dtype=bool)


def _make_sim_data(contacts: np.ndarray, step_dt: float = 0.02) -> dict[str, np.ndarray]:
    num_steps, num_feet = contacts.shape
    foot_positions = np.zeros((num_steps, num_feet, 3), dtype=float)
    foot_positions[:, :, 2] = np.where(contacts, 0.0, 0.04)
    commands = np.zeros((num_steps, 3), dtype=float)
    commands[:, 0] = 1.0
    base_velocity = commands.copy()
    return {
        "sim_times": np.arange(num_steps, dtype=float) * step_dt,
        "contact_forces_array": contacts.astype(float) * 2.0,
        "foot_positions_contact_frame_array": foot_positions,
        "foot_labels": np.asarray(FOOT_LABELS),
        "commanded_velocity_array": commands,
        "base_linear_velocity_array": base_velocity,
    }


def test_contact_metrics_resolve_diagonal_feet_by_label_and_drop_clipped_segments() -> None:
    contacts = _diagonal_contacts([4] * 8)

    metrics = compute_scenario_contact_metrics(
        sim_data=_make_sim_data(contacts),
        analysis_mask=np.ones(len(contacts), dtype=bool),
        foot_labels=FOOT_LABELS,
        step_dt=0.02,
    )

    assert metrics["diagonal_support_occupancy"] == pytest.approx(1.0)
    assert metrics["stance_duration_iqr_s"] == pytest.approx(0.0)
    assert metrics["swing_duration_iqr_s"] == pytest.approx(0.0)
    assert metrics["stance_event_count"] == 12
    assert metrics["swing_event_count"] == 12


def test_modulated_contacts_increase_stance_and_swing_duration_iqr() -> None:
    flat_contacts = _diagonal_contacts([4] * 12)
    uneven_contacts = _diagonal_contacts([2, 4, 6, 8] * 3)

    flat = compute_scenario_contact_metrics(
        _make_sim_data(flat_contacts),
        np.ones(len(flat_contacts), dtype=bool),
        FOOT_LABELS,
        0.02,
    )
    uneven = compute_scenario_contact_metrics(
        _make_sim_data(uneven_contacts),
        np.ones(len(uneven_contacts), dtype=bool),
        FOOT_LABELS,
        0.02,
    )

    assert uneven["stance_duration_iqr_s"] > flat["stance_duration_iqr_s"]
    assert uneven["swing_duration_iqr_s"] > flat["swing_duration_iqr_s"]


def test_source_analysis_pairs_full_flat_and_command_matched_uneven_scenarios(
    tmp_path: Path,
) -> None:
    flat_contacts = _diagonal_contacts([5] * 12)
    uneven_contacts = _diagonal_contacts([2, 3, 5, 4, 7, 9] * 2)
    contacts = np.concatenate([flat_contacts, uneven_contacts], axis=0)
    sim_data = _make_sim_data(contacts, step_dt=0.1)
    eval_dir = tmp_path / "eval_checkpoint_100_seed_46_action_delay_0"
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True)
    np.savez(plots_dir / "sim_data.npz", **sim_data)
    json_path = eval_dir / "metrics_summary.json"
    json_path.write_text("{}", encoding="utf-8")

    flat_tag = "walk_x_flat_terrain_1.0mps"
    uneven_tag = "fast_walk_x_uneven_terrain"
    summary = {
        "env_name": "env",
        "run_name": "2026-01-01-00-00-00",
        "seed": 46,
        "random_sim_steps": 0,
        "total_sim_steps": 120,
        "automatic_reset_steps": [59, 119],
        "fixed_command_scenarios": [
            [flat_tag, [1.0, 0.0, 0.0], None],
            [uneven_tag, [1.0, 0.0, 0.0], None],
        ],
    }
    source = TerrainRunSource(
        label="LEP",
        env_name="env",
        run_name="2026-01-01-00-00-00",
        checkpoint=100,
        json_path=json_path,
        metrics_summary=summary,
    )

    per_scenario, paired, manifest = analyze_terrain_sources(
        [source],
        flat_tag=flat_tag,
        uneven_tag=uneven_tag,
    )

    assert len(per_scenario) == 2
    assert len(paired) == 1
    assert paired.iloc[0]["run_name"] == source.run_name
    assert paired.iloc[0]["uneven_scenario"] == uneven_tag
    assert paired.iloc[0]["stance_duration_iqr_s_delta"] > 0.0
    assert set(per_scenario["command_vx"]) == {1.0}
    assert manifest.iloc[0]["status"] == "analyzed"


def test_source_failure_does_not_leave_an_unpaired_scenario_row(tmp_path: Path) -> None:
    contacts = _diagonal_contacts([5] * 18)
    eval_dir = tmp_path / "eval_checkpoint_100"
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True)
    np.savez(plots_dir / "sim_data.npz", **_make_sim_data(contacts, step_dt=0.1))
    json_path = eval_dir / "metrics_summary.json"
    json_path.write_text("{}", encoding="utf-8")
    flat_tag = "walk_x_flat_terrain_1.0mps"
    uneven_tag = "fast_walk_x_uneven_terrain"
    source = TerrainRunSource(
        label="LEP",
        env_name="env",
        run_name="2026-01-01-00-00-00",
        checkpoint=100,
        json_path=json_path,
        metrics_summary={
            "seed": 46,
            "random_sim_steps": 0,
            "total_sim_steps": 120,
            "automatic_reset_steps": [59, 119],
            "fixed_command_scenarios": [
                [flat_tag, [1.0, 0.0, 0.0], None],
                [uneven_tag, [1.0, 0.0, 0.0], None],
            ],
        },
    )

    per_scenario, paired, manifest = analyze_terrain_sources(
        [source], flat_tag=flat_tag, uneven_tag=uneven_tag
    )

    assert per_scenario.empty
    assert paired.empty
    assert manifest.iloc[0]["status"] == "skipped"
    assert manifest.iloc[0]["warning"]


def _three_run_pairs() -> pd.DataFrame:
    rows = []
    for run_name, occupancy_delta, stance_delta in (
        ("low-run", -0.10, 0.04),
        ("median-run", -0.15, 0.08),
        ("high-run", -0.20, 0.12),
    ):
        rows.append(
            {
                "label": "LEP",
                "env_name": "env",
                "run_name": run_name,
                "diagonal_support_occupancy_flat": 0.75,
                "diagonal_support_occupancy_uneven": 0.75 + occupancy_delta,
                "diagonal_support_occupancy_delta": occupancy_delta,
                "stance_duration_iqr_s_flat": 0.03,
                "stance_duration_iqr_s_uneven": 0.03 + stance_delta,
                "stance_duration_iqr_s_delta": stance_delta,
            }
        )
    return pd.DataFrame(rows)


def test_aggregation_uses_one_paired_value_per_trained_run() -> None:
    summary = aggregate_terrain_adaptation(_three_run_pairs())

    row = summary.loc[
        summary["metric"] == "diagonal_support_occupancy_delta"
    ].iloc[0]
    assert row["n"] == 3
    assert row["mean"] == pytest.approx(-0.15)


def test_representative_run_is_closest_to_median_headline_changes() -> None:
    assert select_representative_run(_three_run_pairs(), "LEP") == "median-run"


def test_output_writer_creates_compact_figure_tables_and_interpretation_report(
    tmp_path: Path,
) -> None:
    flat_contacts = _diagonal_contacts([4] * 8)
    uneven_contacts = _diagonal_contacts([2, 4, 6, 4, 2, 6, 4, 4])
    contacts = np.concatenate([flat_contacts, uneven_contacts])
    sim_data_path = tmp_path / "sim_data.npz"
    np.savez(sim_data_path, **_make_sim_data(contacts))
    split = len(flat_contacts)
    per_scenario = pd.DataFrame(
        [
            {
                "label": "LEP",
                "run_name": "median-run",
                "terrain_condition": "flat",
                "scenario": "walk_x_flat_terrain_1.0mps",
                "sim_data_path": str(sim_data_path),
                "analysis_start_step": 0,
                "analysis_end_step": split,
            },
            {
                "label": "LEP",
                "run_name": "median-run",
                "terrain_condition": "uneven",
                "scenario": "fast_walk_x_uneven_terrain",
                "sim_data_path": str(sim_data_path),
                "analysis_start_step": split,
                "analysis_end_step": len(contacts),
            },
        ]
    )
    manifest = pd.DataFrame(
        [{"label": "LEP", "run_name": "median-run", "status": "analyzed", "warning": ""}]
    )

    paths = write_terrain_adaptation_outputs(
        per_scenario_df=per_scenario,
        paired_df=_three_run_pairs(),
        manifest_df=manifest,
        output_dir=tmp_path / "outputs",
        primary_label="LEP",
        export_formats=["pdf"],
        figure_width=7.0,
        figure_height=4.5,
        grid_alpha=0.3,
    )

    assert Path(paths["per_scenario"]).is_file()
    assert Path(paths["summary"]).is_file()
    assert Path(paths["figure_pdf"]).is_file()
    report = Path(paths["report"]).read_text(encoding="utf-8")
    assert "terrain-conditioned timing modulation" in report
    assert "discrete gait-family transitions" in report
    assert "Duty factor" not in report
    assert "conditional on exactly two" not in report
