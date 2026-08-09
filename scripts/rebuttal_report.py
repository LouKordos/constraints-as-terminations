from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any


REBUTTAL_DYNAMICS_SCENARIOS = (
    "walk_x_flat_terrain_1.0mps",
    "fast_walk_stairs_up",
    "medium_walk_diagonal_turning_uneven_terrain",
    "fast_walk_diagonal_uneven_terrain",
    "fast_walk_x_uneven_terrain"
)

REPORT_PLOT_FILENAMES = (
    "support_dynamics.pdf",
    "base_acceleration.pdf",
    "base_excitation_summary.pdf",
    "vertical_ground_reaction_force.pdf",
    "vertical_ground_reaction_force_distribution.pdf",
    "touchdown_impact_distribution.pdf",
)

CSV_FIELDS = (
    "scenario",
    "status",
    "completed",
    "valid_duration_seconds",
    "sample_count",
    "aerial_phase_percent",
    "low_support_percent",
    "mean_duty_factor",
    "vertical_velocity_rms_m_s",
    "vertical_velocity_mean_abs_m_s",
    "vertical_acceleration_rms_g",
    "vertical_acceleration_mean_abs_g",
    "vertical_acceleration_abs_p95_g",
    "vertical_acceleration_abs_p99_g",
    "pitch_rate_rms_rad_s",
    "pitch_rate_mean_abs_rad_s",
    "pitch_acceleration_rms_rad_s2",
    "pitch_acceleration_mean_abs_rad_s2",
    "total_vertical_grf_p95_body_weight",
    "total_vertical_grf_p99_body_weight",
    "joint_acceleration_rms_rad_s2",
    "joint_acceleration_mean_abs_rad_s2",
    "joint_acceleration_abs_p95_rad_s2",
    "joint_acceleration_abs_p99_rad_s2",
    "linear_velocity_x_rmse_m_s",
    "linear_velocity_x_mae_m_s",
    "linear_velocity_y_rmse_m_s",
    "linear_velocity_y_mae_m_s",
    "cost_of_transport",
    "max_operational_limit_violation_percent",
    "max_operational_limit_violation_frequency_percent",
    "max_operational_limit_violation_constraint",
    "max_operational_limit_violation_element",
    "max_operational_limit_violation_absolute_excess",
    "base_acceleration_fidelity",
    "vertical_grf_fidelity",
)


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _percent(value: Any) -> float | None:
    return None if value is None else float(value) * 100.0


def _scenario_row(scenario: str, entry: dict[str, Any]) -> dict[str, Any]:
    window = entry.get("analysis_window", {})
    dynamics = entry.get("gait_dynamics", {})
    context = entry.get("context", {})
    uses_severity_schema = (
        "max_operational_limit_violation_frequency_percent" in context
        or isinstance(context.get("maximum_constraint_violation"), dict)
    )
    legacy_maximum_frequency = context.get("max_operational_limit_violation_percent")
    return {
        "scenario": scenario,
        "status": "evaluated",
        "completed": window.get("completed"),
        "valid_duration_seconds": window.get("valid_duration_seconds"),
        "sample_count": window.get("sample_count"),
        "aerial_phase_percent": _percent(
            _nested(dynamics, "support_dynamics", "aerial_phase_fraction")
        ),
        "low_support_percent": _percent(
            _nested(dynamics, "support_dynamics", "low_support_fraction")
        ),
        "mean_duty_factor": _nested(dynamics, "support_dynamics", "mean_duty_factor"),
        "vertical_velocity_rms_m_s": _nested(
            dynamics, "base_excitation", "vertical_velocity_rms_m_s"
        ),
        "vertical_velocity_mean_abs_m_s": _nested(
            dynamics, "base_excitation", "vertical_velocity_mean_abs_m_s"
        ),
        "vertical_acceleration_rms_g": _nested(
            dynamics, "base_excitation", "vertical_acceleration_rms_g"
        ),
        "vertical_acceleration_mean_abs_g": _nested(
            dynamics, "base_excitation", "vertical_acceleration_mean_abs_g"
        ),
        "vertical_acceleration_abs_p95_g": _nested(
            dynamics, "base_excitation", "vertical_acceleration_abs_p95_g"
        ),
        "vertical_acceleration_abs_p99_g": _nested(
            dynamics, "base_excitation", "vertical_acceleration_abs_p99_g"
        ),
        "pitch_rate_rms_rad_s": _nested(
            dynamics, "base_excitation", "pitch_rate_rms_rad_s"
        ),
        "pitch_rate_mean_abs_rad_s": _nested(
            dynamics, "base_excitation", "pitch_rate_mean_abs_rad_s"
        ),
        "pitch_acceleration_rms_rad_s2": _nested(
            dynamics, "base_excitation", "pitch_acceleration_rms_rad_s2"
        ),
        "pitch_acceleration_mean_abs_rad_s2": _nested(
            dynamics, "base_excitation", "pitch_acceleration_mean_abs_rad_s2"
        ),
        "total_vertical_grf_p95_body_weight": _nested(
            dynamics, "impact_loading", "total_vertical_grf_body_weight", "abs_p95"
        ),
        "total_vertical_grf_p99_body_weight": _nested(
            dynamics, "impact_loading", "total_vertical_grf_body_weight", "abs_p99"
        ),
        "joint_acceleration_rms_rad_s2": _nested(
            dynamics, "joint_demand", "joint_acceleration", "rms"
        ),
        "joint_acceleration_mean_abs_rad_s2": _nested(
            dynamics, "joint_demand", "joint_acceleration", "mean_abs"
        ),
        "joint_acceleration_abs_p95_rad_s2": _nested(
            dynamics, "joint_demand", "joint_acceleration", "abs_p95"
        ),
        "joint_acceleration_abs_p99_rad_s2": _nested(
            dynamics, "joint_demand", "joint_acceleration", "abs_p99"
        ),
        "linear_velocity_x_rmse_m_s": context.get("base_linear_velocity_x_rms_error"),
        "linear_velocity_x_mae_m_s": context.get(
            "base_linear_velocity_x_mean_abs_error"
        ),
        "linear_velocity_y_rmse_m_s": context.get("base_linear_velocity_y_rms_error"),
        "linear_velocity_y_mae_m_s": context.get(
            "base_linear_velocity_y_mean_abs_error"
        ),
        "cost_of_transport": context.get("cost_of_transport"),
        "max_operational_limit_violation_percent": (
            context.get("max_operational_limit_violation_percent")
            if uses_severity_schema
            else None
        ),
        "max_operational_limit_violation_frequency_percent": (
            context.get("max_operational_limit_violation_frequency_percent")
            if uses_severity_schema
            else legacy_maximum_frequency
        ),
        "max_operational_limit_violation_constraint": context.get(
            "max_operational_limit_violation_constraint"
        ),
        "max_operational_limit_violation_element": context.get(
            "max_operational_limit_violation_element"
        ),
        "max_operational_limit_violation_absolute_excess": context.get(
            "max_operational_limit_violation_absolute_excess"
        ),
        "base_acceleration_fidelity": _nested(
            dynamics, "data_fidelity", "base_acceleration"
        ),
        "vertical_grf_fidelity": _nested(dynamics, "data_fidelity", "vertical_grf"),
    }


def build_rebuttal_payload(
    summary_metrics: dict[str, Any],
    plot_status: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Extract a stable, paper-oriented subset from the complete evaluation summary."""
    available = summary_metrics.get("fixed_command_scenarios_gait_dynamics_metrics", {})
    scenarios: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for scenario in REBUTTAL_DYNAMICS_SCENARIOS:
        if scenario not in available:
            scenarios[scenario] = {"status": "not evaluated"}
            warnings.append(f"Scenario `{scenario}` was not evaluated.")
            continue

        entry = available[scenario]
        scenarios[scenario] = {"status": "evaluated", **entry}
        row = _scenario_row(scenario, entry)
        rows.append(row)

        window = entry.get("analysis_window", {})
        if not window.get("completed", False):
            warnings.append(
                f"Scenario `{scenario}` reset prematurely at step "
                f"{window.get('premature_reset_step')}; its dynamics window was truncated."
            )
        if int(window.get("sample_count", 0)) == 0:
            warnings.append(f"Scenario `{scenario}` has no valid post-warm-up dynamics samples.")

        if row["base_acceleration_fidelity"] != "direct":
            warnings.append(
                f"Scenario `{scenario}` uses `{row['base_acceleration_fidelity']}` base acceleration."
            )
        if row["vertical_grf_fidelity"] != "direct_vector_history":
            warnings.append(
                f"Scenario `{scenario}` uses `{row['vertical_grf_fidelity']}` vertical GRF."
            )

    normalized_plot_status = {str(key): int(value) for key, value in (plot_status or {}).items()}
    for plot_name, return_code in normalized_plot_status.items():
        if return_code != 0:
            warnings.append(f"Plot job `{plot_name}` plot generation returned exit code {return_code}.")

    return {
        "report_version": 1,
        "requested_scenarios": list(REBUTTAL_DYNAMICS_SCENARIOS),
        "analysis_protocol": {
            "warmup_seconds": 1.0,
            "window": "Exclude the first 1.0 s and stop before the first automatic reset.",
            "contact_threshold_newtons": 1.0,
        },
        "run": {
            "run_dir": summary_metrics.get("run_dir"),
            "checkpoint_path": summary_metrics.get("used_checkpoint_path"),
            "task_name": summary_metrics.get("task_name"),
            "seed": summary_metrics.get("seed"),
            "action_delay_steps": summary_metrics.get("action_delay_steps"),
        },
        "scenarios": scenarios,
        "rows": rows,
        "plot_status": normalized_plot_status,
        "warnings": warnings,
    }


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _format_constraint_identity(row: dict[str, Any]) -> str:
    constraint = row.get("max_operational_limit_violation_constraint")
    element = row.get("max_operational_limit_violation_element")
    if constraint is None:
        return "N/A"
    return str(constraint) if element is None else f"{constraint}:{element}"


def render_rebuttal_markdown(payload: dict[str, Any]) -> str:
    """Render the focused payload as a self-contained Markdown report."""
    run = payload["run"]
    lines = [
        "# Rebuttal Gait-Dynamics Evaluation Report",
        "",
        f"- Run: `{run.get('run_dir')}`",
        f"- Checkpoint: `{run.get('checkpoint_path')}`",
        f"- Task: `{run.get('task_name')}`",
        f"- Seed: `{run.get('seed')}`",
        f"- Action delay: `{run.get('action_delay_steps')}` policy steps",
        "",
        "## Analysis protocol",
        "",
        "Each fixed scenario is analyzed separately. The first 1.0 s is excluded as gait-initiation "
        "warm-up, and the window stops before the first automatic reset. Contact uses a strict "
        "resultant-force threshold of 1.0 N.",
        "",
        "## Headline dynamics metrics",
        "",
        "| Scenario | Complete | Valid (s) | Aerial (%) | <=1 support (%) | Duty factor | "
        "Vertical velocity RMS / mean abs (m/s) | Vertical acceleration RMS / mean abs (g) | "
        "GRF p95 / p99 (BW) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in payload["requested_scenarios"]:
        entry = payload["scenarios"][scenario]
        if entry["status"] != "evaluated":
            lines.append(f"| `{scenario}` | not evaluated | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        row = next(row for row in payload["rows"] if row["scenario"] == scenario)
        lines.append(
            f"| `{scenario}` | {_format(row['completed'])} | {_format(row['valid_duration_seconds'])} | "
            f"{_format(row['aerial_phase_percent'], 2)} | {_format(row['low_support_percent'], 2)} | "
            f"{_format(row['mean_duty_factor'])} | {_format(row['vertical_velocity_rms_m_s'])} / "
            f"{_format(row['vertical_velocity_mean_abs_m_s'])} | "
            f"{_format(row['vertical_acceleration_rms_g'])} / "
            f"{_format(row['vertical_acceleration_mean_abs_g'])} | "
            f"{_format(row['total_vertical_grf_p95_body_weight'])} / "
            f"{_format(row['total_vertical_grf_p99_body_weight'])} |"
        )

    lines.extend(
        [
            "",
            "## Supporting transfer metrics",
            "",
            "Joint acceleration is included as supporting actuator-bandwidth evidence; the existing "
            "per-joint acceleration plots remain the detailed source and are not duplicated here.",
            "",
            "| Scenario | Pitch rate RMS / mean abs (rad/s) | "
            "Pitch acceleration RMS / mean abs (rad/s^2) | "
            "Joint acceleration RMS / mean abs / p95 / p99 (rad/s^2) | "
            "vx RMSE / MAE | vy RMSE / MAE | COT | "
            "Max normalized excess (%) / source | Max exceedance frequency (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| `{row['scenario']}` | {_format(row['pitch_rate_rms_rad_s'])} / "
            f"{_format(row['pitch_rate_mean_abs_rad_s'])} | "
            f"{_format(row['pitch_acceleration_rms_rad_s2'])} / "
            f"{_format(row['pitch_acceleration_mean_abs_rad_s2'])} | "
            f"{_format(row['joint_acceleration_rms_rad_s2'])} / "
            f"{_format(row['joint_acceleration_mean_abs_rad_s2'])} / "
            f"{_format(row['joint_acceleration_abs_p95_rad_s2'])} / "
            f"{_format(row['joint_acceleration_abs_p99_rad_s2'])} | "
            f"{_format(row['linear_velocity_x_rmse_m_s'])} / "
            f"{_format(row['linear_velocity_x_mae_m_s'])} | "
            f"{_format(row['linear_velocity_y_rmse_m_s'])} / "
            f"{_format(row['linear_velocity_y_mae_m_s'])} | {_format(row['cost_of_transport'])} | "
            f"{_format(row['max_operational_limit_violation_percent'])} / "
            f"{_format_constraint_identity(row)} | "
            f"{_format(row['max_operational_limit_violation_frequency_percent'])} |"
        )

    lines.extend(["", "## Plots", ""])
    for row in payload["rows"]:
        scenario = row["scenario"]
        lines.append(f"### `{scenario}`")
        lines.append("")
        for filename in REPORT_PLOT_FILENAMES:
            relative = f"plots/scenario_{scenario}/rebuttal/{filename}"
            lines.append(f"- [{filename}]({relative})")
        lines.extend(
            [
                f"- Existing joint acceleration plots: `plots/scenario_{scenario}/joint_metrics/acceleration/`",
                f"- Existing gait diagram: `plots/scenario_{scenario}/aggregates/gait_diagram.pdf`",
                "",
            ]
        )

    lines.extend(["## Warnings", ""])
    if payload["warnings"]:
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These measurements test whether the policy uses more aerial, body-exciting, and "
            "impact-loaded locomotion. Together with the controlled LP--LEP ablation they support a "
            "mechanistic sim-to-real explanation, but they do not by themselves prove universal "
            "causality across robots or tasks.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as output_file:
            output_file.write(content)
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def write_rebuttal_reports(
    summary_metrics: dict[str, Any],
    output_dir: str,
    plot_status: dict[str, int] | None = None,
) -> dict[str, str]:
    """Write focused Markdown, JSON, and CSV reports and return their paths."""
    output_path = Path(output_dir)
    payload = build_rebuttal_payload(summary_metrics, plot_status=plot_status)

    markdown_path = output_path / "rebuttal_gait_dynamics_report.md"
    json_path = output_path / "rebuttal_gait_dynamics_metrics.json"
    csv_path = output_path / "rebuttal_gait_dynamics_metrics.csv"

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in payload["rows"]:
        writer.writerow({field: row.get(field) for field in CSV_FIELDS})

    _atomic_write_text(markdown_path, render_rebuttal_markdown(payload))
    _atomic_write_text(json_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _atomic_write_text(csv_path, csv_buffer.getvalue())

    return {
        "markdown": str(markdown_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }
