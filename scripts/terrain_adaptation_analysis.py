from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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
