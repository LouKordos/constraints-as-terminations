from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rebuttal_report import REBUTTAL_DYNAMICS_SCENARIOS, build_rebuttal_payload


SCENARIO_DISPLAY_NAMES = {
    "walk_x_flat_terrain_1.0mps": "Flat 1.0 m/s",
    "fast_walk_stairs_up": "Stairs up",
    "medium_walk_diagonal_turning_uneven_terrain": "Diagonal + turn",
    "fast_walk_diagonal_uneven_terrain": "Fast diagonal",
}

AGGREGATE_METRICS = (
    "completion_percent",
    "valid_duration_seconds",
    "aerial_phase_percent",
    "low_support_percent",
    "mean_duty_factor",
    "contact_transitions_per_second",
    "vertical_velocity_rms_m_s",
    "vertical_acceleration_rms_g",
    "vertical_acceleration_abs_p95_g",
    "vertical_acceleration_abs_p99_g",
    "pitch_rate_rms_rad_s",
    "pitch_acceleration_rms_rad_s2",
    "total_vertical_grf_p95_body_weight",
    "total_vertical_grf_p99_body_weight",
    "joint_acceleration_rms_rad_s2",
    "joint_acceleration_abs_p95_rad_s2",
    "joint_acceleration_abs_p99_rad_s2",
    "linear_velocity_x_rmse_m_s",
    "linear_velocity_y_rmse_m_s",
    "cost_of_transport",
    "max_operational_limit_violation_percent",
)

METRIC_DISPLAY_NAMES = {
    "completion_percent": "Completion rate (%)",
    "valid_duration_seconds": "Valid duration (s)",
    "aerial_phase_percent": "Aerial time (%)",
    "low_support_percent": "At most one foot supporting (%)",
    "mean_duty_factor": "Mean duty factor",
    "contact_transitions_per_second": "Contact transitions (s$^{-1}$)",
    "vertical_velocity_rms_m_s": "Vertical velocity RMS (m/s)",
    "vertical_acceleration_rms_g": "Vertical acceleration RMS (g)",
    "vertical_acceleration_abs_p95_g": "Vertical acceleration p95 (g)",
    "vertical_acceleration_abs_p99_g": "Vertical acceleration p99 (g)",
    "pitch_rate_rms_rad_s": "Pitch rate RMS (rad/s)",
    "pitch_acceleration_rms_rad_s2": "Pitch acceleration RMS (rad/s$^2$)",
    "total_vertical_grf_p95_body_weight": "Vertical GRF p95 (BW)",
    "total_vertical_grf_p99_body_weight": "Vertical GRF p99 (BW)",
    "joint_acceleration_rms_rad_s2": "Joint acceleration RMS (rad/s$^2$)",
    "joint_acceleration_abs_p95_rad_s2": "Joint acceleration p95 (rad/s$^2$)",
    "joint_acceleration_abs_p99_rad_s2": "Joint acceleration p99 (rad/s$^2$)",
    "linear_velocity_x_rmse_m_s": "$v_x$ RMSE (m/s)",
    "linear_velocity_y_rmse_m_s": "$v_y$ RMSE (m/s)",
    "cost_of_transport": "Cost of transport",
    "max_operational_limit_violation_percent": "Max. operational-limit violation (%)",
}

PLOT_FAMILIES = {
    "plot_rebuttal_headline_dynamics": (
        "aerial_phase_percent",
        "vertical_acceleration_rms_g",
        "total_vertical_grf_p95_body_weight",
        "joint_acceleration_rms_rad_s2",
    ),
    "plot_rebuttal_support_dynamics": (
        "aerial_phase_percent",
        "low_support_percent",
        "mean_duty_factor",
        "contact_transitions_per_second",
    ),
    "plot_rebuttal_base_excitation": (
        "vertical_velocity_rms_m_s",
        "vertical_acceleration_rms_g",
        "pitch_rate_rms_rad_s",
        "pitch_acceleration_rms_rad_s2",
    ),
    "plot_rebuttal_impact_and_joint_demand": (
        "total_vertical_grf_p95_body_weight",
        "total_vertical_grf_p99_body_weight",
        "joint_acceleration_rms_rad_s2",
        "joint_acceleration_abs_p95_rad_s2",
    ),
    "plot_rebuttal_completion_rate": ("completion_percent",),
}

PLOT_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


@dataclass(frozen=True)
class EvaluationMetadata:
    checkpoint: int | None
    action_delay_steps: int
    evaluation_scope: str


@dataclass(frozen=True)
class RebuttalGaitRunData:
    env_name: str
    run_name: str
    checkpoint: int | None
    action_delay_steps: int
    evaluation_scope: str
    json_path: Path
    metrics_summary: dict[str, Any]


def _extract_checkpoint_with_suffix(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.match(r"^eval_checkpoint_(\d+)(?:_.*)?$", part)
        if match:
            return int(match.group(1))
    return None


def extract_evaluation_metadata(
    summary: dict[str, Any],
    json_path: Path,
) -> EvaluationMetadata:
    raw_action_delay = summary.get("action_delay_steps")
    if raw_action_delay is None:
        path_match = re.search(r"_action_delay_(\d+)(?:_|$)", str(json_path))
        action_delay_steps = int(path_match.group(1)) if path_match else 0
    else:
        action_delay_steps = int(raw_action_delay)
    evaluation_scope = (
        "rebuttal_only" if summary.get("rebuttal_scenarios_only") is True else "full"
    )
    return EvaluationMetadata(
        checkpoint=_extract_checkpoint_with_suffix(json_path),
        action_delay_steps=action_delay_steps,
        evaluation_scope=evaluation_scope,
    )


def _is_rebuttal_gait_summary(summary: dict[str, Any]) -> bool:
    dynamics = summary.get("fixed_command_scenarios_gait_dynamics_metrics")
    return isinstance(dynamics, dict) and bool(dynamics)


def _has_current_evaluation_metadata(entry: RebuttalGaitRunData) -> bool:
    return "action_delay_steps" in entry.metrics_summary


def _choose_preferred_gait_entry(
    key: tuple[str, str],
    entries: list[RebuttalGaitRunData],
) -> tuple[RebuttalGaitRunData | None, str | None]:
    eligible = [
        entry
        for entry in entries
        if entry.action_delay_steps == 0 and entry.checkpoint is not None
    ]
    if not eligible:
        return None, None

    full_entries = [entry for entry in eligible if entry.evaluation_scope == "full"]
    preferred_scope = full_entries if full_entries else eligible
    max_checkpoint = max(
        entry.checkpoint for entry in preferred_scope if entry.checkpoint is not None
    )
    latest = [entry for entry in preferred_scope if entry.checkpoint == max_checkpoint]

    if len(latest) > 1:
        current = [entry for entry in latest if _has_current_evaluation_metadata(entry)]
        if current:
            latest = current
    if len(latest) != 1:
        raise ValueError(
            "Multiple equally preferred gait-dynamics summaries found for "
            f"env_name={key[0]!r}, run_name={key[1]!r}: "
            f"{[str(entry.json_path) for entry in latest]}"
        )

    selected = latest[0]
    if selected.evaluation_scope == "full":
        reason = "highest_checkpoint_full_zero_delay"
    else:
        reason = "highest_checkpoint_rebuttal_only_fallback"
    return selected, reason


def discover_rebuttal_gait_dynamics_files(
    root_dirs: Iterable[Path],
) -> tuple[dict[tuple[str, str], RebuttalGaitRunData], pd.DataFrame]:
    """Discover full or purpose-built zero-delay evaluations containing gait metrics."""
    candidates: dict[tuple[str, str], list[RebuttalGaitRunData]] = {}
    manifest_rows: list[dict[str, Any]] = []
    visited_paths: set[Path] = set()

    for root_dir in root_dirs:
        for json_path in sorted(Path(root_dir).rglob("metrics_summary.json")):
            resolved_path = json_path.resolve()
            if resolved_path in visited_paths:
                continue
            visited_paths.add(resolved_path)

            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
            except (OSError, json.JSONDecodeError) as exception:
                logging.warning("Skipping unreadable evaluation summary %s: %s", json_path, exception)
                continue

            if not isinstance(summary, dict) or not _is_rebuttal_gait_summary(summary):
                continue

            env_name = summary.get("env_name")
            run_name = summary.get("run_name")
            metadata = extract_evaluation_metadata(summary, json_path)
            if not isinstance(env_name, str) or not env_name or not isinstance(run_name, str) or not run_name:
                logging.warning("Skipping gait summary without valid env/run identity: %s", json_path)
                continue

            entry = RebuttalGaitRunData(
                env_name=env_name,
                run_name=run_name,
                checkpoint=metadata.checkpoint,
                action_delay_steps=metadata.action_delay_steps,
                evaluation_scope=metadata.evaluation_scope,
                json_path=json_path,
                metrics_summary=summary,
            )
            candidates.setdefault((env_name, run_name), []).append(entry)
            manifest_rows.append(
                {
                    "env_name": env_name,
                    "run_name": run_name,
                    "checkpoint": metadata.checkpoint,
                    "eval_seed": summary.get("seed"),
                    "action_delay_steps": metadata.action_delay_steps,
                    "evaluation_scope": metadata.evaluation_scope,
                    "eligible": metadata.action_delay_steps == 0 and metadata.checkpoint is not None,
                    "selected": False,
                    "selection_reason": (
                        "excluded_nonzero_action_delay"
                        if metadata.action_delay_steps != 0
                        else "excluded_unparseable_checkpoint"
                        if metadata.checkpoint is None
                        else "not_selected"
                    ),
                    "json_path": str(json_path),
                }
            )

    selected: dict[tuple[str, str], RebuttalGaitRunData] = {}
    selected_reasons: dict[Path, str] = {}
    for key, entries in sorted(candidates.items()):
        selected_entry, reason = _choose_preferred_gait_entry(key, entries)
        if selected_entry is None or reason is None:
            continue
        selected[key] = selected_entry
        selected_reasons[selected_entry.json_path.resolve()] = reason

    manifest = pd.DataFrame(manifest_rows)
    if not manifest.empty:
        resolved_manifest_paths = manifest["json_path"].map(lambda value: Path(value).resolve())
        manifest["selected"] = resolved_manifest_paths.isin(selected_reasons)
        manifest.loc[manifest["selected"], "selection_reason"] = resolved_manifest_paths[
            manifest["selected"]
        ].map(selected_reasons)
        manifest.loc[
            manifest["eligible"] & ~manifest["selected"], "selection_reason"
        ] = "lower_priority_candidate"
        manifest.sort_values(["env_name", "run_name", "checkpoint", "json_path"], inplace=True)
        manifest.reset_index(drop=True, inplace=True)
    return selected, manifest


def filter_rebuttal_runs_for_series(
    index: dict[tuple[str, str], RebuttalGaitRunData],
    env_name: str,
    start_dt: Any,
    end_dt: Any,
    parse_run_datetime,
    is_within_range,
) -> dict[str, RebuttalGaitRunData]:
    selected: dict[str, RebuttalGaitRunData] = {}
    for (entry_env_name, run_name), entry in index.items():
        if entry_env_name != env_name:
            continue
        run_dt = parse_run_datetime(run_name)
        if run_dt is None or not is_within_range(run_dt, start_dt, end_dt):
            continue
        selected[run_name] = entry
    return selected


def _summary_from_entry(entry: RebuttalGaitRunData | dict[str, Any]) -> tuple[dict[str, Any], Any, Any]:
    if isinstance(entry, RebuttalGaitRunData):
        return entry.metrics_summary, entry.checkpoint, str(entry.json_path)
    return entry, entry.get("checkpoint"), entry.get("json_path")


def build_rebuttal_per_run_dataframe(
    series_runs: dict[str, dict[str, RebuttalGaitRunData | dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenario_order = {scenario: index for index, scenario in enumerate(REBUTTAL_DYNAMICS_SCENARIOS)}

    for label, runs in series_runs.items():
        for run_name, entry in sorted(runs.items()):
            summary, checkpoint, json_path = _summary_from_entry(entry)
            payload = build_rebuttal_payload(summary)
            raw_scenarios = summary.get("fixed_command_scenarios_gait_dynamics_metrics", {})
            for row in payload["rows"]:
                scenario = row["scenario"]
                support = (
                    raw_scenarios.get(scenario, {})
                    .get("gait_dynamics", {})
                    .get("support_dynamics", {})
                )
                completed = row.get("completed")
                rows.append(
                    {
                        "label": label,
                        "env_name": summary.get("env_name"),
                        "run_name": summary.get("run_name", run_name),
                        "checkpoint": checkpoint,
                        "eval_seed": summary.get("seed"),
                        "action_delay_steps": summary.get("action_delay_steps"),
                        "json_path": json_path,
                        "scenario": scenario,
                        "scenario_order": scenario_order.get(scenario, len(scenario_order)),
                        **row,
                        "completion_percent": 100.0 if completed is True else 0.0,
                        "contact_transitions_per_second": support.get(
                            "contact_transitions_per_second"
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame()
    dataframe = pd.DataFrame(rows)
    dataframe.sort_values(["label", "run_name", "scenario_order"], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def _metric_stats(values: pd.Series) -> dict[str, Any]:
    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(cleaned))
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "sem": np.nan, "ci95": np.nan, "n": 0}
    mean = float(cleaned.mean())
    if n == 1:
        return {"mean": mean, "std": np.nan, "sem": np.nan, "ci95": np.nan, "n": 1}
    std = float(cleaned.std(ddof=1))
    sem = std / np.sqrt(n)
    return {"mean": mean, "std": std, "sem": sem, "ci95": 1.96 * sem, "n": n}


def _aggregate_long(
    dataframe: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, group in dataframe.groupby(group_columns, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identity = dict(zip(group_columns, group_key))
        for metric in AGGREGATE_METRICS:
            if metric not in group.columns:
                continue
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "display_name": METRIC_DISPLAY_NAMES.get(metric, metric),
                    **_metric_stats(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_rebuttal_gait_dynamics(
    per_run_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if per_run_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    per_scenario = _aggregate_long(
        per_run_df,
        ["label", "env_name", "scenario", "scenario_order"],
    )

    identity_columns = [
        column
        for column in ("label", "env_name", "run_name", "checkpoint", "eval_seed", "action_delay_steps")
        if column in per_run_df.columns
    ]
    numeric_columns = [metric for metric in AGGREGATE_METRICS if metric in per_run_df.columns]
    overall_per_run = (
        per_run_df.groupby(identity_columns, dropna=False, as_index=False)[numeric_columns]
        .mean(numeric_only=True)
    )
    overall = _aggregate_long(overall_per_run, ["label", "env_name"])
    return per_scenario, overall_per_run, overall


def _format_ci(mean: Any, ci95: Any, n: Any, digits: int = 3) -> str:
    if pd.isna(mean) or int(n) == 0:
        return "N/A"
    if int(n) == 1 or pd.isna(ci95):
        return f"{float(mean):.{digits}f} (n={int(n)})"
    return f"{float(mean):.{digits}f} ± {float(ci95):.{digits}f} (95% CI, n={int(n)})"


def _render_aggregate_report(
    per_run_df: pd.DataFrame,
    per_scenario_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> str:
    lines = [
        "# Aggregate Rebuttal Gait-Dynamics Report",
        "",
        "## Seed-level aggregation",
        "",
        "Each timestamped training run is one independent seed. Metrics are first reduced to one "
        "value per run and scenario. Overall values average the four scenarios within each run "
        "before computing the across-run mean and 95% confidence interval.",
        "",
    ]
    for label, group in per_run_df.groupby("label", sort=False):
        eval_seeds = sorted(pd.to_numeric(group["eval_seed"], errors="coerce").dropna().astype(int).unique())
        lines.append(
            f"- **{label}:** {group['run_name'].nunique()} trained runs; "
            f"evaluation seed(s) {eval_seeds}; {len(group)} run-scenario rows."
        )

    lines.extend(
        [
            "",
            "## Overall headline metrics",
            "",
            "| Variant | Aerial time (%) | Vertical acceleration RMS (g) | "
            "Vertical GRF p95 (BW) | Joint acceleration RMS (rad/s²) | Completion (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    headline = (
        "aerial_phase_percent",
        "vertical_acceleration_rms_g",
        "total_vertical_grf_p95_body_weight",
        "joint_acceleration_rms_rad_s2",
        "completion_percent",
    )
    for label in per_run_df["label"].drop_duplicates():
        cells = []
        for metric in headline:
            match = overall_df[(overall_df["label"] == label) & (overall_df["metric"] == metric)]
            if match.empty:
                cells.append("N/A")
            else:
                row = match.iloc[0]
                cells.append(_format_ci(row["mean"], row["ci95"], row["n"]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Per-scenario headline metrics", ""])
    for scenario in REBUTTAL_DYNAMICS_SCENARIOS:
        lines.extend(
            [
                f"### {SCENARIO_DISPLAY_NAMES.get(scenario, scenario)}",
                "",
                "| Variant | Aerial (%) | Vertical acceleration RMS (g) | GRF p95 (BW) | Joint acceleration RMS | Completion (%) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label in per_run_df["label"].drop_duplicates():
            cells = []
            for metric in headline:
                match = per_scenario_df[
                    (per_scenario_df["label"] == label)
                    & (per_scenario_df["scenario"] == scenario)
                    & (per_scenario_df["metric"] == metric)
                ]
                if match.empty:
                    cells.append("N/A")
                else:
                    row = match.iloc[0]
                    cells.append(_format_ci(row["mean"], row["ci95"], row["n"]))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(["## Data-quality warnings", ""])
    warnings: list[str] = []
    expected_runs = len(per_run_df[["label", "run_name"]].drop_duplicates())
    expected_rows = expected_runs * len(REBUTTAL_DYNAMICS_SCENARIOS)
    if len(per_run_df) != expected_rows:
        warnings.append(
            f"Expected {expected_rows} run-scenario rows but found {len(per_run_df)}; at least one scenario is missing."
        )
    incomplete = per_run_df[per_run_df["completion_percent"] < 100.0]
    if not incomplete.empty:
        warnings.append(f"{len(incomplete)} run-scenario evaluations reset before scenario completion.")
    non_direct_accel = per_run_df[per_run_df["base_acceleration_fidelity"] != "direct"]
    if not non_direct_accel.empty:
        warnings.append(f"{len(non_direct_accel)} rows do not use direct base acceleration.")
    non_direct_grf = per_run_df[per_run_df["vertical_grf_fidelity"] != "direct_vector_history"]
    if not non_direct_grf.empty:
        warnings.append(f"{len(non_direct_grf)} rows do not use direct vector-history vertical GRF.")
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Generated plots",
            "",
            *[f"- `{stem}.pdf`" for stem in PLOT_FAMILIES],
            "",
            "The points show individual trained runs; bars/markers and whiskers show the across-run mean and 95% CI.",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_family_figure_size(
    metric_count: int,
    requested_width: float,
    requested_height: float,
) -> tuple[float, float]:
    if metric_count <= 1:
        return requested_width, requested_height
    return max(7.0, requested_width), max(4.4, requested_height)


def _plot_metric_family(
    per_run_df: pd.DataFrame,
    metrics: tuple[str, ...],
    output_path_stem: Path,
    export_formats: list[str],
    figure_width: float,
    figure_height: float,
    grid_alpha: float,
) -> list[Path]:
    labels = per_run_df["label"].drop_duplicates().tolist()
    scenarios = [scenario for scenario in REBUTTAL_DYNAMICS_SCENARIOS if scenario in set(per_run_df["scenario"])]
    columns = 2 if len(metrics) > 1 else 1
    rows = int(np.ceil(len(metrics) / columns))
    resolved_width, resolved_height = _resolve_family_figure_size(
        metric_count=len(metrics),
        requested_width=figure_width,
        requested_height=figure_height,
    )
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(resolved_width, resolved_height),
        squeeze=False,
        layout="constrained",
    )
    flat_axes = axes.ravel()
    x_positions = np.arange(len(scenarios), dtype=float)
    total_width = 0.72
    label_width = total_width / max(1, len(labels))

    for metric_index, metric in enumerate(metrics):
        ax = flat_axes[metric_index]
        for label_index, label in enumerate(labels):
            offset = (label_index - (len(labels) - 1) / 2.0) * label_width
            means: list[float] = []
            ci_values: list[float] = []
            for scenario_index, scenario in enumerate(scenarios):
                values = pd.to_numeric(
                    per_run_df[
                        (per_run_df["label"] == label) & (per_run_df["scenario"] == scenario)
                    ][metric],
                    errors="coerce",
                ).dropna()
                stats = _metric_stats(values)
                means.append(stats["mean"])
                ci_values.append(0.0 if pd.isna(stats["ci95"]) else stats["ci95"])
                if not values.empty:
                    jitter = np.linspace(-0.22, 0.22, len(values)) * label_width
                    ax.scatter(
                        np.full(len(values), x_positions[scenario_index] + offset) + jitter,
                        values,
                        s=11,
                        alpha=0.45,
                        color=PLOT_COLORS[label_index % len(PLOT_COLORS)],
                        linewidths=0,
                        zorder=3,
                    )
            ax.errorbar(
                x_positions + offset,
                means,
                yerr=ci_values,
                fmt="o",
                markersize=4.5,
                capsize=2.5,
                linewidth=1.0,
                color=PLOT_COLORS[label_index % len(PLOT_COLORS)],
                label=label,
                zorder=4,
            )
        ax.set_ylabel(METRIC_DISPLAY_NAMES.get(metric, metric))
        ax.set_xticks(x_positions)
        ax.set_xticklabels([SCENARIO_DISPLAY_NAMES.get(scenario, scenario) for scenario in scenarios])
        ax.tick_params(axis="x", labelrotation=20)
        for tick_label in ax.get_xticklabels():
            tick_label.set_ha("right")
        ax.grid(True, axis="y", color="0.82", linewidth=0.5, alpha=grid_alpha)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric_index == 0:
            ax.legend(frameon=False, ncol=min(3, len(labels)))

    for unused_axis in flat_axes[len(metrics) :]:
        unused_axis.set_visible(False)
    paths: list[Path] = []
    for export_format in export_formats:
        path = output_path_stem.with_suffix(f".{export_format}")
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def write_rebuttal_aggregate_outputs(
    per_run_df: pd.DataFrame,
    output_dir: Path,
    export_formats: list[str],
    figure_width: float,
    figure_height: float,
    ci_alpha: float,
    grid_alpha: float,
    plot_style: str,
) -> dict[str, str]:
    del ci_alpha, plot_style  # The seed points and error bars encode uncertainty directly.
    if per_run_df.empty:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_scenario, overall_per_run, overall = aggregate_rebuttal_gait_dynamics(per_run_df)
    tables = {
        "per_run": output_dir / "rebuttal_gait_dynamics_per_run.csv",
        "per_scenario": output_dir / "rebuttal_gait_dynamics_per_scenario.csv",
        "overall_per_run": output_dir / "rebuttal_gait_dynamics_overall_per_run.csv",
        "overall": output_dir / "rebuttal_gait_dynamics_overall.csv",
    }
    per_run_df.to_csv(tables["per_run"], index=False)
    per_scenario.to_csv(tables["per_scenario"], index=False)
    overall_per_run.to_csv(tables["overall_per_run"], index=False)
    overall.to_csv(tables["overall"], index=False)

    report_path = output_dir / "rebuttal_gait_dynamics_report.md"
    report_path.write_text(
        _render_aggregate_report(per_run_df, per_scenario, overall),
        encoding="utf-8",
    )

    paths: dict[str, str] = {key: str(path) for key, path in tables.items()}
    paths["report"] = str(report_path)
    for stem, metrics in PLOT_FAMILIES.items():
        plot_paths = _plot_metric_family(
            per_run_df=per_run_df,
            metrics=metrics,
            output_path_stem=output_dir / stem,
            export_formats=export_formats,
            figure_width=figure_width,
            figure_height=figure_height,
            grid_alpha=grid_alpha,
        )
        for path in plot_paths:
            paths[f"{stem}_{path.suffix.lstrip('.')}"] = str(path)
    return paths
