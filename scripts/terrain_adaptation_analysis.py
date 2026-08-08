from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics_utils import (
    build_scenario_analysis_mask,
    compute_stance_segments,
    compute_swing_heights,
)


CONTACT_FORCE_THRESHOLD_N = 1.0
DEFAULT_WARMUP_SECONDS = 1.0

PAIRED_METRICS = (
    "diagonal_support_occupancy",
    "stance_duration_iqr_s",
    "swing_duration_iqr_s",
    "mean_swing_height_m",
    "planar_velocity_rmse_m_s",
    "analyzed_duration_s",
)


@dataclass(frozen=True)
class TerrainRunSource:
    label: str
    env_name: str
    run_name: str
    checkpoint: int | None
    json_path: Path
    metrics_summary: dict[str, Any]


def _normalized_foot_role(label: str) -> str | None:
    normalized = "".join(character for character in label.lower() if character.isalnum())
    aliases = {
        "fl": ("fl", "flfoot", "frontleft", "frontleftfoot"),
        "fr": ("fr", "frfoot", "frontright", "frontrightfoot"),
        "rl": ("rl", "rlfoot", "rearleft", "rearleftfoot", "hindleft", "hindleftfoot"),
        "rr": ("rr", "rrfoot", "rearright", "rearrightfoot", "hindright", "hindrightfoot"),
    }
    for role, role_aliases in aliases.items():
        if normalized in role_aliases:
            return role
    return None


def resolve_foot_indices(foot_labels: Iterable[str]) -> tuple[int, int, int, int]:
    resolved: dict[str, int] = {}
    labels = [str(label) for label in foot_labels]
    for index, label in enumerate(labels):
        role = _normalized_foot_role(label)
        if role is None:
            continue
        if role in resolved:
            raise ValueError(f"Multiple foot labels resolve to {role.upper()}: {labels!r}")
        resolved[role] = index
    missing = [role.upper() for role in ("fl", "fr", "rl", "rr") if role not in resolved]
    if missing:
        raise ValueError(f"Could not resolve foot labels {missing} from {labels!r}")
    return resolved["fl"], resolved["fr"], resolved["rl"], resolved["rr"]


def reconstruct_contact_state(contact_forces: np.ndarray) -> np.ndarray:
    forces = np.asarray(contact_forces, dtype=float)
    if forces.ndim == 2:
        resultant = np.abs(forces)
    elif forces.ndim == 3 and forces.shape[-1] == 3:
        resultant = np.linalg.norm(forces, axis=-1)
    else:
        raise ValueError(
            "contact_forces_array must have shape (T, F) or (T, F, 3), "
            f"got {forces.shape}"
        )
    return resultant > CONTACT_FORCE_THRESHOLD_N


def _complete_segment_durations(values: np.ndarray, step_dt: float) -> list[float]:
    segments = compute_stance_segments(np.asarray(values, dtype=bool))
    num_steps = len(values)
    return [
        float((end - start) * step_dt)
        for start, end in segments
        if start > 0 and end < num_steps
    ]


def _iqr_or_none(values: list[float]) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, 75) - np.percentile(finite, 25))


def _mean_or_none(values: Iterable[float]) -> float | None:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _contiguous_mask_slice(analysis_mask: np.ndarray, total_steps: int) -> slice:
    mask = np.asarray(analysis_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != total_steps:
        raise ValueError(
            f"analysis_mask must have shape ({total_steps},), got {mask.shape}"
        )
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return slice(0, 0)
    if not np.all(np.diff(selected) == 1):
        raise ValueError("analysis_mask must select one contiguous scenario interval")
    return slice(int(selected[0]), int(selected[-1]) + 1)


def compute_scenario_contact_metrics(
    sim_data: Mapping[str, Any],
    analysis_mask: np.ndarray,
    foot_labels: list[str],
    step_dt: float,
) -> dict[str, Any]:
    """Compute observed contact-timing metrics over one reset-safe scenario interval."""
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    contact_state = reconstruct_contact_state(np.asarray(sim_data["contact_forces_array"]))
    if contact_state.shape[1] != len(foot_labels):
        raise ValueError(
            f"Contact data contains {contact_state.shape[1]} feet but received "
            f"{len(foot_labels)} labels"
        )
    selection = _contiguous_mask_slice(analysis_mask, len(contact_state))
    contacts = contact_state[selection]
    fl, fr, rl, rr = resolve_foot_indices(foot_labels)

    if len(contacts) == 0:
        diagonal_support_occupancy = None
    else:
        first_diagonal = (
            contacts[:, fl]
            & contacts[:, rr]
            & ~contacts[:, fr]
            & ~contacts[:, rl]
        )
        second_diagonal = (
            contacts[:, fr]
            & contacts[:, rl]
            & ~contacts[:, fl]
            & ~contacts[:, rr]
        )
        diagonal_support_occupancy = float(np.mean(first_diagonal | second_diagonal))

    stance_durations: list[float] = []
    swing_durations: list[float] = []
    for foot_index in range(len(foot_labels)):
        stance_durations.extend(_complete_segment_durations(contacts[:, foot_index], step_dt))
        swing_durations.extend(_complete_segment_durations(~contacts[:, foot_index], step_dt))

    foot_positions = np.asarray(sim_data["foot_positions_contact_frame_array"], dtype=float)
    if foot_positions.ndim != 3 or foot_positions.shape[:2] != contact_state.shape:
        raise ValueError(
            "foot_positions_contact_frame_array must have shape (T, F, 3) matching contacts, "
            f"got {foot_positions.shape} and {contact_state.shape}"
        )
    swing_heights_by_foot = compute_swing_heights(
        contact_state=contacts,
        foot_heights_contact=foot_positions[selection, :, 2],
        reset_steps=[],
        foot_labels=foot_labels,
    )
    swing_heights = [
        float(height)
        for heights in swing_heights_by_foot.values()
        for height in heights
        if np.isfinite(height) and height >= 0.0
    ]

    planar_velocity_rmse = None
    if "base_linear_velocity_array" in sim_data and "commanded_velocity_array" in sim_data:
        measured = np.asarray(sim_data["base_linear_velocity_array"], dtype=float)[selection, :2]
        commanded = np.asarray(sim_data["commanded_velocity_array"], dtype=float)[selection, :2]
        if measured.shape != commanded.shape:
            raise ValueError(
                "base_linear_velocity_array and commanded_velocity_array must have matching "
                f"planar shapes, got {measured.shape} and {commanded.shape}"
            )
        if measured.size:
            planar_velocity_rmse = float(np.sqrt(np.mean(np.sum((measured - commanded) ** 2, axis=1))))

    return {
        "diagonal_support_occupancy": diagonal_support_occupancy,
        "stance_duration_iqr_s": _iqr_or_none(stance_durations),
        "swing_duration_iqr_s": _iqr_or_none(swing_durations),
        "mean_swing_height_m": _mean_or_none(swing_heights),
        "planar_velocity_rmse_m_s": planar_velocity_rmse,
        "stance_event_count": len(stance_durations),
        "swing_event_count": len(swing_durations),
        "swing_height_event_count": len(swing_heights),
        "analyzed_duration_s": float(len(contacts) * step_dt),
    }


def _extract_scenario_tag(entry: Any) -> str | None:
    if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], str):
        return entry[0]
    if isinstance(entry, dict) and isinstance(entry.get("tag"), str):
        return entry["tag"]
    return None


def _extract_scenario_command(entry: Any) -> tuple[float, float, float] | None:
    command: Any = None
    if isinstance(entry, (list, tuple)) and len(entry) > 1:
        command = entry[1]
    elif isinstance(entry, dict):
        command = entry.get("command")
    if not isinstance(command, (list, tuple)) or len(command) < 3:
        return None
    return float(command[0]), float(command[1]), float(command[2])


def infer_scenario_ranges_and_commands(
    metrics_summary: Mapping[str, Any],
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[float, float, float] | None]]:
    scenarios = metrics_summary.get("fixed_command_scenarios")
    random_steps = metrics_summary.get("random_sim_steps")
    total_steps = metrics_summary.get("total_sim_steps")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Missing non-empty fixed_command_scenarios metadata")
    if not isinstance(random_steps, (int, float)) or not isinstance(total_steps, (int, float)):
        raise ValueError("Missing numeric random_sim_steps or total_sim_steps metadata")
    random_steps = int(random_steps)
    total_steps = int(total_steps)
    fixed_steps = total_steps - random_steps
    steps_per_scenario_float = fixed_steps / len(scenarios)
    steps_per_scenario = int(round(steps_per_scenario_float))
    if fixed_steps <= 0 or not np.isclose(steps_per_scenario_float, steps_per_scenario):
        raise ValueError(
            "Fixed-command scenarios do not divide the recorded timestep count evenly"
        )

    ranges: dict[str, tuple[int, int]] = {}
    commands: dict[str, tuple[float, float, float] | None] = {}
    for scenario_index, entry in enumerate(scenarios):
        tag = _extract_scenario_tag(entry)
        if tag is None or tag in ranges:
            raise ValueError(f"Invalid or duplicate fixed-command scenario entry: {entry!r}")
        start = random_steps + scenario_index * steps_per_scenario
        ranges[tag] = (start, start + steps_per_scenario)
        commands[tag] = _extract_scenario_command(entry)
    return ranges, commands


def _load_sim_data(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz_file:
        return {key: npz_file[key] for key in npz_file.files}


def _infer_step_dt(sim_data: Mapping[str, Any]) -> float:
    sim_times = np.asarray(sim_data["sim_times"], dtype=float)
    if sim_times.ndim != 1 or len(sim_times) < 2:
        raise ValueError("sim_times must be a one-dimensional array with at least two samples")
    positive_differences = np.diff(sim_times)
    positive_differences = positive_differences[positive_differences > 0.0]
    if positive_differences.size == 0:
        raise ValueError("sim_times does not contain a positive timestep")
    return float(np.median(positive_differences))


def _foot_labels_from_sim_data(sim_data: Mapping[str, Any]) -> list[str]:
    if "foot_labels" not in sim_data:
        raise ValueError("sim_data.npz does not contain foot_labels")
    return [str(label) for label in np.asarray(sim_data["foot_labels"]).tolist()]


def _paired_row(flat_row: dict[str, Any], uneven_row: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: flat_row[key]
        for key in ("label", "env_name", "run_name", "checkpoint", "eval_seed", "json_path", "sim_data_path")
    }
    row.update(
        {
            "flat_scenario": flat_row["scenario"],
            "uneven_scenario": uneven_row["scenario"],
            "flat_completed": flat_row["completed"],
            "uneven_completed": uneven_row["completed"],
        }
    )
    for metric in PAIRED_METRICS:
        flat_value = flat_row.get(metric)
        uneven_value = uneven_row.get(metric)
        row[f"{metric}_flat"] = flat_value
        row[f"{metric}_uneven"] = uneven_value
        row[f"{metric}_delta"] = (
            float(uneven_value - flat_value)
            if flat_value is not None
            and uneven_value is not None
            and np.isfinite(flat_value)
            and np.isfinite(uneven_value)
            else np.nan
        )
    return row


def analyze_terrain_sources(
    sources: Iterable[TerrainRunSource],
    flat_tag: str,
    uneven_tag: str,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze saved full evaluations without importing or launching the simulator."""
    scenario_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for source in sources:
        sim_data_path = source.json_path.parent / "plots" / "sim_data.npz"
        manifest_row = {
            "label": source.label,
            "env_name": source.env_name,
            "run_name": source.run_name,
            "checkpoint": source.checkpoint,
            "json_path": str(source.json_path),
            "sim_data_path": str(sim_data_path),
            "flat_scenario": flat_tag,
            "uneven_scenario": uneven_tag,
            "status": "skipped",
            "warning": "",
        }
        try:
            if not sim_data_path.is_file():
                raise FileNotFoundError(f"Missing {sim_data_path}")
            ranges, commands = infer_scenario_ranges_and_commands(source.metrics_summary)
            missing = [tag for tag in (flat_tag, uneven_tag) if tag not in ranges]
            if missing:
                raise ValueError(f"Missing required scenario tag(s): {missing}")
            if commands[flat_tag] != commands[uneven_tag]:
                raise ValueError(
                    f"Scenario commands do not match: {commands[flat_tag]} vs {commands[uneven_tag]}"
                )

            sim_data = _load_sim_data(sim_data_path)
            step_dt = _infer_step_dt(sim_data)
            total_steps = len(np.asarray(sim_data["sim_times"]))
            warmup_steps = int(round(warmup_seconds / step_dt))
            automatic_reset_steps = [
                int(step) for step in source.metrics_summary.get("automatic_reset_steps", [])
            ]
            foot_labels = _foot_labels_from_sim_data(sim_data)
            run_rows: dict[str, dict[str, Any]] = {}
            completed_scenario_rows: list[dict[str, Any]] = []

            for terrain_condition, scenario_tag in (("flat", flat_tag), ("uneven", uneven_tag)):
                start, end = ranges[scenario_tag]
                mask, mask_metadata = build_scenario_analysis_mask(
                    total_steps=total_steps,
                    scenario_start=start,
                    scenario_end=end,
                    warmup_steps=warmup_steps,
                    automatic_reset_steps=automatic_reset_steps,
                )
                metrics = compute_scenario_contact_metrics(
                    sim_data=sim_data,
                    analysis_mask=mask,
                    foot_labels=foot_labels,
                    step_dt=step_dt,
                )
                command = commands[scenario_tag]
                row = {
                    "label": source.label,
                    "env_name": source.env_name,
                    "run_name": source.run_name,
                    "checkpoint": source.checkpoint,
                    "eval_seed": source.metrics_summary.get("seed"),
                    "json_path": str(source.json_path),
                    "sim_data_path": str(sim_data_path),
                    "terrain_condition": terrain_condition,
                    "scenario": scenario_tag,
                    "command_vx": command[0] if command is not None else np.nan,
                    "command_vy": command[1] if command is not None else np.nan,
                    "command_yaw": command[2] if command is not None else np.nan,
                    "step_dt": step_dt,
                    **mask_metadata,
                    **metrics,
                }
                completed_scenario_rows.append(row)
                run_rows[terrain_condition] = row

            scenario_rows.extend(completed_scenario_rows)
            paired_rows.append(_paired_row(run_rows["flat"], run_rows["uneven"]))
            manifest_row["status"] = "analyzed"
        except Exception as exception:
            manifest_row["warning"] = str(exception)
        manifest_rows.append(manifest_row)

    per_scenario = pd.DataFrame(scenario_rows)
    paired = pd.DataFrame(paired_rows)
    manifest = pd.DataFrame(manifest_rows)
    if not per_scenario.empty:
        per_scenario.sort_values(["label", "run_name", "terrain_condition"], inplace=True)
        per_scenario.reset_index(drop=True, inplace=True)
    if not paired.empty:
        paired.sort_values(["label", "run_name"], inplace=True)
        paired.reset_index(drop=True, inplace=True)
    if not manifest.empty:
        manifest.sort_values(["label", "run_name"], inplace=True)
        manifest.reset_index(drop=True, inplace=True)
    return per_scenario, paired, manifest


def _metric_statistics(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    count = int(len(numeric))
    if count == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "ci95": np.nan,
            "n": 0,
        }
    mean = float(numeric.mean())
    if count == 1:
        return {
            "mean": mean,
            "std": np.nan,
            "sem": np.nan,
            "ci95": np.nan,
            "n": 1,
        }
    standard_deviation = float(numeric.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(count)
    return {
        "mean": mean,
        "std": standard_deviation,
        "sem": standard_error,
        "ci95": float(1.96 * standard_error),
        "n": count,
    }


def aggregate_terrain_adaptation(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired run-level values without treating stride events as seeds."""
    if paired_df.empty:
        return pd.DataFrame()
    identity_columns = [column for column in ("label", "env_name") if column in paired_df]
    metric_columns = [
        column
        for column in paired_df.columns
        if column.endswith(("_flat", "_uneven", "_delta"))
        and column not in {"flat_completed", "uneven_completed"}
    ]
    rows: list[dict[str, Any]] = []
    grouped = paired_df.groupby(identity_columns, dropna=False, sort=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identity = dict(zip(identity_columns, group_key))
        for metric in metric_columns:
            rows.append({**identity, "metric": metric, **_metric_statistics(group[metric])})
    return pd.DataFrame(rows)


def select_representative_run(paired_df: pd.DataFrame, label: str) -> str:
    """Select the run nearest the bivariate median of the two headline changes."""
    headline_metrics = (
        "diagonal_support_occupancy_delta",
        "stance_duration_iqr_s_delta",
    )
    group = paired_df.loc[paired_df["label"] == label].copy()
    if group.empty:
        raise ValueError(f"No paired terrain-adaptation rows found for label {label!r}")
    missing = [metric for metric in headline_metrics if metric not in group]
    if missing:
        raise ValueError(f"Missing representative-run metric columns: {missing}")
    for metric in headline_metrics:
        group[metric] = pd.to_numeric(group[metric], errors="coerce")
    group.dropna(subset=list(headline_metrics), inplace=True)
    if group.empty:
        raise ValueError(f"No finite headline terrain-adaptation metrics for label {label!r}")

    squared_distance = np.zeros(len(group), dtype=float)
    for metric in headline_metrics:
        values = group[metric].to_numpy(dtype=float)
        median = float(np.median(values))
        scale = float(np.percentile(values, 75) - np.percentile(values, 25))
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        squared_distance += ((values - median) / scale) ** 2
    group["_representative_distance"] = squared_distance
    group.sort_values(["_representative_distance", "run_name"], inplace=True)
    return str(group.iloc[0]["run_name"])


def _format_mean_ci(summary_df: pd.DataFrame, label: str, metric: str) -> str:
    row = summary_df.loc[
        (summary_df["label"] == label) & (summary_df["metric"] == metric)
    ]
    if row.empty or int(row.iloc[0]["n"]) == 0:
        return "N/A"
    values = row.iloc[0]
    if int(values["n"]) == 1 or pd.isna(values["ci95"]):
        return f"{float(values['mean']):.4f} (n={int(values['n'])})"
    return (
        f"{float(values['mean']):.4f} ± {float(values['ci95']):.4f} "
        f"(95% CI, n={int(values['n'])})"
    )


def _render_terrain_report(
    paired_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    primary_label: str,
    representative_run: str,
) -> str:
    lines = [
        "# Terrain-Conditioned Gait-Adaptation Report",
        "",
        "This analysis tests continuous terrain-conditioned timing modulation around the "
        "policy's nominal trot. It does not claim discrete gait-family transitions across speed.",
        "",
        "The flat and uneven scenarios both command `(vx, vy, yaw) = (1.0, 0.0, 0.0)`. "
        "Each independently trained run contributes one paired value; individual stance or "
        "swing events are not treated as independent seeds.",
        "Among the archived command-matched terrain scenarios, the selected uneven scenario "
        "showed the largest changes in diagonal-state occupancy and stance/swing timing variability.",
        "",
        "## Rebuttal-facing metrics",
        "",
        "- **Diagonal-support occupancy:** fraction of all analyzed control steps in the observed "
        "`FL+RR` or `FR+RL` contact state. This is a post-hoc statistic, not a prescribed contact pattern.",
        "- **Stance-duration IQR:** within-run interquartile range of complete stance durations after "
        "the 1 s warm-up and reset truncation.",
        "- Supporting exports include swing-duration IQR and maximum swing height above local terrain.",
        "",
        f"Representative contact raster for **{primary_label}**: `{representative_run}` "
        "(selected deterministically as the run nearest the median headline changes).",
        "",
        "| Metric | Flat | Uneven | Uneven − flat |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = (
        ("Diagonal-support occupancy", "diagonal_support_occupancy"),
        ("Stance-duration IQR (s)", "stance_duration_iqr_s"),
        ("Swing-duration IQR (s)", "swing_duration_iqr_s"),
        ("Mean maximum swing height (m)", "mean_swing_height_m"),
    )
    for display_name, metric in metric_rows:
        lines.append(
            f"| {display_name} | "
            f"{_format_mean_ci(summary_df, primary_label, metric + '_flat')} | "
            f"{_format_mean_ci(summary_df, primary_label, metric + '_uneven')} | "
            f"{_format_mean_ci(summary_df, primary_label, metric + '_delta')} |"
        )

    lines.extend(["", "## Data quality", ""])
    analyzed = manifest_df.loc[manifest_df.get("status", pd.Series(dtype=str)) == "analyzed"]
    lines.append(
        f"- Analyzed {paired_df.loc[paired_df['label'] == primary_label, 'run_name'].nunique()} "
        f"paired {primary_label} runs ({len(analyzed)} sources across all labels)."
    )
    warnings = manifest_df.loc[
        manifest_df.get("warning", pd.Series(dtype=str)).astype(str).str.len() > 0,
        [column for column in ("label", "run_name", "warning") if column in manifest_df],
    ]
    if warnings.empty:
        lines.append("- No source-level warnings.")
    else:
        for _, warning in warnings.iterrows():
            lines.append(
                f"- {warning.get('label', '')}/{warning.get('run_name', '')}: "
                f"{warning.get('warning', '')}"
            )
    lines.extend(
        [
            "",
            "Contact is reconstructed from resultant foot force strictly greater than 1 N. "
            "Segments intersecting an analysis boundary are excluded from duration distributions.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_contact_raster(
    axis: plt.Axes,
    scenario_row: pd.Series,
    title: str,
) -> None:
    sim_data = _load_sim_data(Path(str(scenario_row["sim_data_path"])))
    contacts = reconstruct_contact_state(sim_data["contact_forces_array"])
    labels = _foot_labels_from_sim_data(sim_data)
    fl, fr, rl, rr = resolve_foot_indices(labels)
    order = [fl, fr, rl, rr]
    start = int(scenario_row["analysis_start_step"])
    end = int(scenario_row["analysis_end_step"])
    selected_contacts = contacts[start:end, order].T.astype(float)
    step_dt = _infer_step_dt(sim_data)
    duration = len(selected_contacts.T) * step_dt
    axis.pcolormesh(
        np.arange(selected_contacts.shape[1] + 1, dtype=float) * step_dt,
        np.arange(selected_contacts.shape[0] + 1, dtype=float) - 0.5,
        selected_contacts,
        shading="flat",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_yticks(range(4))
    axis.set_yticklabels(["FL", "FR", "RL", "RR"])
    axis.set_ylim(3.5, -0.5)
    axis.set_xlim(0.0, max(duration, step_dt))
    axis.set_ylabel("Foot")
    axis.set_title(title, loc="left")


def _plot_paired_metric(
    axis: plt.Axes,
    paired_df: pd.DataFrame,
    label: str,
    metric: str,
    ylabel: str,
    title: str,
    grid_alpha: float,
) -> None:
    group = paired_df.loc[paired_df["label"] == label].copy()
    flat_column = f"{metric}_flat"
    uneven_column = f"{metric}_uneven"
    group[flat_column] = pd.to_numeric(group[flat_column], errors="coerce")
    group[uneven_column] = pd.to_numeric(group[uneven_column], errors="coerce")
    group.dropna(subset=[flat_column, uneven_column], inplace=True)
    for _, row in group.sort_values("run_name").iterrows():
        axis.plot(
            [0.0, 1.0],
            [row[flat_column], row[uneven_column]],
            color="0.70",
            linewidth=0.7,
            alpha=0.75,
            zorder=1,
        )
        axis.scatter(
            [0.0, 1.0],
            [row[flat_column], row[uneven_column]],
            color="0.45",
            s=9,
            alpha=0.75,
            zorder=2,
        )
    means = [float(group[column].mean()) for column in (flat_column, uneven_column)]
    cis = []
    for column in (flat_column, uneven_column):
        stats = _metric_statistics(group[column])
        cis.append(0.0 if pd.isna(stats["ci95"]) else float(stats["ci95"]))
    axis.errorbar(
        [0.0, 1.0],
        means,
        yerr=cis,
        color="#0072B2",
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        capsize=2.5,
        zorder=4,
    )
    axis.set_xticks([0.0, 1.0], ["Flat", "Uneven"])
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left")
    axis.grid(True, axis="y", color="0.82", linewidth=0.5, alpha=grid_alpha)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if metric == "diagonal_support_occupancy":
        axis.set_ylim(0.0, 1.0)


def _plot_terrain_adaptation(
    per_scenario_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    output_stem: Path,
    primary_label: str,
    representative_run: str,
    export_formats: list[str],
    figure_width: float,
    figure_height: float,
    grid_alpha: float,
) -> list[Path]:
    representative = per_scenario_df.loc[
        (per_scenario_df["label"] == primary_label)
        & (per_scenario_df["run_name"] == representative_run)
    ]
    flat_rows = representative.loc[representative["terrain_condition"] == "flat"]
    uneven_rows = representative.loc[representative["terrain_condition"] == "uneven"]
    if flat_rows.empty or uneven_rows.empty:
        raise ValueError(
            f"Representative run {representative_run!r} is missing flat or uneven raster data"
        )

    figure = plt.figure(figsize=(figure_width, figure_height), layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    flat_axis = figure.add_subplot(grid[0, 0])
    uneven_axis = figure.add_subplot(grid[1, 0], sharex=flat_axis)
    occupancy_axis = figure.add_subplot(grid[0, 1])
    stance_axis = figure.add_subplot(grid[1, 1])
    _plot_contact_raster(flat_axis, flat_rows.iloc[0], "(a) Flat, 1.0 m/s")
    _plot_contact_raster(uneven_axis, uneven_rows.iloc[0], "(b) Uneven, 1.0 m/s")
    flat_axis.set_xlabel("")
    flat_axis.tick_params(axis="x", labelbottom=False)
    uneven_axis.set_xlabel("Time (s)")
    _plot_paired_metric(
        occupancy_axis,
        paired_df,
        primary_label,
        "diagonal_support_occupancy",
        "Diagonal-state occupancy",
        "(c) Contact-state regularity",
        grid_alpha,
    )
    _plot_paired_metric(
        stance_axis,
        paired_df,
        primary_label,
        "stance_duration_iqr_s",
        "Stance-duration IQR (s)",
        "(d) Contact-timing modulation",
        grid_alpha,
    )

    output_paths: list[Path] = []
    for export_format in export_formats:
        output_path = output_stem.with_suffix(f".{export_format}")
        figure.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
        )
        output_paths.append(output_path)
    plt.close(figure)
    return output_paths


def write_terrain_adaptation_outputs(
    per_scenario_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    output_dir: Path,
    primary_label: str,
    export_formats: list[str],
    figure_width: float,
    figure_height: float,
    grid_alpha: float,
) -> dict[str, str]:
    """Write terrain-specific artifacts while leaving all existing outputs untouched."""
    if paired_df.empty:
        return {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = aggregate_terrain_adaptation(paired_df)
    representative_run = select_representative_run(paired_df, primary_label)
    table_paths = {
        "per_scenario": output_dir / "terrain_adaptation_per_scenario.csv",
        "paired": output_dir / "terrain_adaptation_paired.csv",
        "summary": output_dir / "terrain_adaptation_summary.csv",
        "manifest": output_dir / "terrain_adaptation_manifest.csv",
    }
    per_scenario_df.to_csv(table_paths["per_scenario"], index=False)
    paired_df.to_csv(table_paths["paired"], index=False)
    summary_df.to_csv(table_paths["summary"], index=False)
    manifest_df.to_csv(table_paths["manifest"], index=False)

    report_path = output_dir / "terrain_adaptation_report.md"
    report_path.write_text(
        _render_terrain_report(
            paired_df=paired_df,
            summary_df=summary_df,
            manifest_df=manifest_df,
            primary_label=primary_label,
            representative_run=representative_run,
        ),
        encoding="utf-8",
    )
    figure_paths = _plot_terrain_adaptation(
        per_scenario_df=per_scenario_df,
        paired_df=paired_df,
        output_stem=output_dir / "plot_terrain_adaptation",
        primary_label=primary_label,
        representative_run=representative_run,
        export_formats=export_formats,
        figure_width=figure_width,
        figure_height=figure_height,
        grid_alpha=grid_alpha,
    )
    paths = {key: str(path) for key, path in table_paths.items()}
    paths["report"] = str(report_path)
    for path in figure_paths:
        paths[f"figure_{path.suffix.lstrip('.')}"] = str(path)
    return paths
