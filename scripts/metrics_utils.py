from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

__all__ = [
    "compute_histogram",
    "compute_trimmed_histogram_data",
    "total_variation_distance",
    "optimal_bin_edges",
    "compute_energy_arrays",
    "compute_stance_segments",
    "summarize_metric",
    "compute_swing_durations",
    "compute_swing_heights",
    "compute_swing_lengths",
    "summarize_dynamics",
    "build_scenario_analysis_mask",
    "compute_support_dynamics",
    "compute_vertical_grf_arrays",
    "compute_vertical_grf_dynamics",
    "compute_gait_dynamics_metrics",
    "compute_summary_metrics"
]


def compute_histogram(arr: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Given a 1D numpy array `arr` and a shared 1D array of bin_edges of length B+1,
    returns a normalized histogram vector of length B (summing to 1.0).
    """
    counts, _ = np.histogram(arr, bins=bin_edges)
    total = counts.sum()
    if total > 0:
        return counts.astype(np.float64) / float(total)
    else:
        # If the array was empty or all zeros, return uniform or zeros.
        # Here we return zeros, so that TVD with another zero‐histogram is 0.
        return np.zeros_like(counts, dtype=np.float64)


def compute_trimmed_histogram_data(data: np.ndarray, bins='auto', lower_percentile=1, upper_percentile=99):
    lower, upper = np.percentile(data, [lower_percentile, upper_percentile])
    data_trimmed = data[(data >= lower) & (data <= upper)]
    return np.histogram(data_trimmed, bins=bins)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Given two 1D numpy vectors p and q (same length, non-negative, each summing to 1),
    compute TVD = 0.5 * sum |p_i - q_i|.
    """
    return 0.5 * np.sum(np.abs(p - q))


def optimal_bin_edges(samples: np.ndarray, rule: str = "fd", min_bins: int = 20, max_bins: int = 250) -> np.ndarray:
    """
    Return a 1-D array of bin edges derived from `samples`
    using either 'fd' (Freedman–Diaconis) or 'scott'.
    """
    n = samples.size
    data_min, data_max = samples.min(), samples.max()
    data_range = data_max - data_min

    if rule == "fd":
        iqr = np.subtract(*np.percentile(samples, [75, 25]))
        h = 2.0 * iqr / np.cbrt(n) if iqr > 0 else None
    elif rule == "scott":
        h = 3.5 * samples.std(ddof=1) / np.cbrt(n)
    else:
        raise ValueError("rule must be 'fd' or 'scott'")

    if h is None or h <= 0:
        h = 3.5 * samples.std(ddof=1) / np.cbrt(n)   # Scott fallback

    num_bins = int(np.clip(np.ceil(data_range / h), min_bins, max_bins))
    return np.linspace(data_min, data_max, num_bins + 1)


def compute_stance_segments(in_contact: np.ndarray) -> list[tuple[int, int]]:
    """
    Return a list of (start_idx, end_idx) indices for every **contact/stance**
    segment of a single foot.  `in_contact` is a 1-D Boolean array over time.
    """
    segments, start = [], None
    for t, val in enumerate(in_contact):
        if val and start is None:
            start = t
        elif not val and start is not None:
            segments.append((start, t)) # [start, end)
            start = None
    if start is not None: # hanging segment
        segments.append((start, len(in_contact)))
    return segments


def compute_swing_segments(in_contact: np.ndarray) -> list[tuple[int, int]]:
    # Air-time segments are contact segments of the inverted array
    return compute_stance_segments(in_contact=~in_contact)

def compute_energy_arrays(power_array: np.ndarray, base_lin_vel: np.ndarray, reset_steps: List[int], step_dt: float, robot_mass: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (energy_per_joint, combined_energy, cot_time_series) for the *full* run.
    """
    instantaneous_speed = np.linalg.norm(base_lin_vel[:, :2], axis=1)

    # repair teleports / resets
    for r in reset_steps:
        instantaneous_speed[max(0, r - 1):r + 1] = instantaneous_speed[max(0, r - 1)]
        power_array[max(0, r - 1):r + 1, :] = power_array[max(0, r - 1), :]

    energy_per_joint = np.cumsum(np.abs(power_array), axis=0) * step_dt
    combined_energy = np.cumsum(np.abs(power_array).sum(axis=1)) * step_dt
    with np.errstate(divide="ignore", invalid="ignore"):
        cost_of_transport_time_series = np.abs(power_array).sum(axis=1) / (robot_mass * 9.81 * instantaneous_speed + 1e-12)
    return energy_per_joint, combined_energy, cost_of_transport_time_series


def summarize_metric(values: list[float]) -> dict[str, float]:
    """Return mean/min/max/… for a list, filling zeros if empty."""
    if not values:
        return {k: 0.0 for k in ("mean", "min", "max", "median", "90th_percentile", "99th_percentile", "stddev")}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "mean_of_abs": float(np.sum(np.abs(arr))) / float(len(values)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "90th_percentile": float(np.percentile(arr, 90)),
        "99th_percentile": float(np.percentile(arr, 99)),
        "stddev": float(arr.std()),
    }


def summarize_dynamics(values: np.ndarray) -> dict[str, float | int | None]:
    """Return finite-only dynamics statistics without fabricating values for empty data."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "stddev": None,
            "rms": None,
            "mean_abs": None,
            "min": None,
            "max": None,
            "median": None,
            "abs_p90": None,
            "abs_p95": None,
            "abs_p99": None,
        }

    absolute = np.abs(finite)
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "stddev": float(finite.std()),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "mean_abs": float(absolute.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "median": float(np.median(finite)),
        "abs_p90": float(np.percentile(absolute, 90)),
        "abs_p95": float(np.percentile(absolute, 95)),
        "abs_p99": float(np.percentile(absolute, 99)),
    }


def build_scenario_analysis_mask(
    total_steps: int,
    scenario_start: int,
    scenario_end: int,
    warmup_steps: int,
    automatic_reset_steps: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a warm-up-trimmed ``[start, end)`` mask ending before the first reset."""
    if total_steps < 0:
        raise ValueError(f"total_steps must be non-negative, got {total_steps}.")
    if not 0 <= scenario_start <= scenario_end <= total_steps:
        raise ValueError(
            "Scenario bounds must satisfy "
            f"0 <= start <= end <= total_steps, got {scenario_start}, {scenario_end}, {total_steps}."
        )
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    resets_in_scenario = sorted(
        int(step) for step in automatic_reset_steps if scenario_start <= int(step) < scenario_end
    )
    expected_timeout_step = scenario_end - 1 if scenario_end > scenario_start else None
    timeout_reset_step = (
        expected_timeout_step if expected_timeout_step is not None and expected_timeout_step in resets_in_scenario else None
    )
    premature_resets = [
        step for step in resets_in_scenario if expected_timeout_step is None or step < expected_timeout_step
    ]
    premature_reset_step = premature_resets[0] if premature_resets else None

    analysis_start = min(scenario_start + warmup_steps, scenario_end)
    if premature_reset_step is not None:
        analysis_end = premature_reset_step
    elif timeout_reset_step is not None:
        analysis_end = timeout_reset_step
    else:
        analysis_end = scenario_end
    analysis_end = max(analysis_start, analysis_end)

    mask = np.zeros(total_steps, dtype=bool)
    mask[analysis_start:analysis_end] = True
    metadata = {
        "scenario_start_step": int(scenario_start),
        "scenario_end_step": int(scenario_end),
        "analysis_start_step": int(analysis_start),
        "analysis_end_step": int(analysis_end),
        "sample_count": int(mask.sum()),
        "completed": premature_reset_step is None,
        "premature_reset_step": premature_reset_step,
        "timeout_reset_step": timeout_reset_step,
    }
    return mask, metadata


def compute_support_dynamics(
    contact_state: np.ndarray,
    step_dt: float,
    foot_labels: list[str],
) -> dict[str, Any]:
    """Summarize support topology, duty factors, and aerial durations."""
    contacts = np.asarray(contact_state, dtype=bool)
    if contacts.ndim != 2:
        raise ValueError(f"contact_state must have shape (T, F), got {contacts.shape}.")
    if contacts.shape[1] != len(foot_labels):
        raise ValueError(
            f"contact_state has {contacts.shape[1]} feet but {len(foot_labels)} labels were provided."
        )
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}.")

    num_steps = contacts.shape[0]
    support_count = contacts.sum(axis=1)
    if num_steps == 0:
        support_fraction = {str(count): None for count in range(len(foot_labels) + 1)}
        duty_factor_per_foot = {label: None for label in foot_labels}
        aerial_fraction = None
        low_support_fraction = None
        mean_duty_factor = None
        transition_rate = None
        aerial_durations: list[float] = []
    else:
        support_fraction = {
            str(count): float(np.mean(support_count == count)) for count in range(len(foot_labels) + 1)
        }
        duty_values = contacts.mean(axis=0)
        duty_factor_per_foot = {
            label: float(duty_values[index]) for index, label in enumerate(foot_labels)
        }
        aerial_fraction = float(np.mean(support_count == 0))
        low_support_fraction = float(np.mean(support_count <= 1))
        mean_duty_factor = float(duty_values.mean())
        transition_count = int(np.count_nonzero(np.diff(contacts.astype(np.int8), axis=0)))
        transition_rate = float(transition_count / (num_steps * step_dt))
        aerial_durations = [
            float((end - start) * step_dt)
            for start, end in compute_stance_segments(support_count == 0)
        ]

    return {
        "sample_count": int(num_steps),
        "aerial_phase_fraction": aerial_fraction,
        "low_support_fraction": low_support_fraction,
        "support_count_fraction": support_fraction,
        "duty_factor_per_foot": duty_factor_per_foot,
        "mean_duty_factor": mean_duty_factor,
        "contact_transitions_per_second": transition_rate,
        "aerial_duration_seconds": summarize_dynamics(np.asarray(aerial_durations, dtype=np.float64)),
    }


def compute_vertical_grf_arrays(
    force_history_w: np.ndarray,
    robot_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return peak-within-control-step total and per-foot positive vertical GRF in body weights."""
    forces = np.asarray(force_history_w, dtype=np.float64)
    if forces.ndim != 4 or forces.shape[-1] != 3:
        raise ValueError(f"force_history_w must have shape (T, H, F, 3), got {forces.shape}.")
    if forces.shape[1] == 0:
        raise ValueError("force_history_w must contain at least one history sample per control step.")
    if robot_mass <= 0.0:
        raise ValueError(f"robot_mass must be positive, got {robot_mass}.")

    body_weight = robot_mass * 9.81
    positive_vertical = np.clip(forces[..., 2], 0.0, None)
    total_vertical_by_history = positive_vertical.sum(axis=2)
    total_vertical_per_control_step = total_vertical_by_history.max(axis=1) / body_weight
    per_foot_peak_per_control_step = positive_vertical.max(axis=1) / body_weight
    return total_vertical_per_control_step, per_foot_peak_per_control_step


def compute_vertical_grf_dynamics(
    force_history_w: np.ndarray,
    contact_state: np.ndarray,
    robot_mass: float,
    foot_labels: list[str],
) -> dict[str, Any]:
    """Summarize synchronized vertical loading and false-to-true touchdown peaks."""
    contacts = np.asarray(contact_state, dtype=bool)
    total_grf_bw, per_foot_peak_bw = compute_vertical_grf_arrays(force_history_w, robot_mass)
    if contacts.ndim != 2 or contacts.shape != per_foot_peak_bw.shape:
        raise ValueError(
            "contact_state must match force-history time and foot dimensions, got "
            f"{contacts.shape} and {per_foot_peak_bw.shape}."
        )
    if contacts.shape[1] != len(foot_labels):
        raise ValueError(
            f"contact_state has {contacts.shape[1]} feet but {len(foot_labels)} labels were provided."
        )

    touchdown = np.zeros_like(contacts, dtype=bool)
    if contacts.shape[0] > 1:
        touchdown[1:] = (~contacts[:-1]) & contacts[1:]

    touchdown_summaries = {}
    touchdown_counts = {}
    for foot_index, label in enumerate(foot_labels):
        values = per_foot_peak_bw[touchdown[:, foot_index], foot_index]
        touchdown_summaries[label] = summarize_dynamics(values)
        touchdown_counts[label] = int(values.size)

    return {
        "total_vertical_grf_body_weight": summarize_dynamics(total_grf_bw),
        "touchdown_peak_vertical_grf_body_weight_per_foot": touchdown_summaries,
        "touchdown_count_per_foot": touchdown_counts,
    }


def _summarize_vector_axes(values: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected a vector array with shape (T, 3), got {array.shape}.")
    return {
        axis: summarize_dynamics(array[:, index])
        for index, axis in enumerate(("x", "y", "z"))
    }


def compute_gait_dynamics_metrics(
    mask: np.ndarray,
    data_arrays: Dict[str, np.ndarray],
    constants: Dict[str, Any],
    data_fidelity: dict[str, str],
) -> dict[str, Any]:
    """Compute the focused gait-dynamics metrics over a reset-safe analysis mask."""
    selection = np.asarray(mask, dtype=bool)
    if selection.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got {selection.shape}.")
    step_dt = float(constants["step_dt"])
    foot_labels = list(constants["foot_labels"])
    robot_mass = float(constants["total_robot_mass"])

    def selected(name: str) -> np.ndarray:
        values = np.asarray(data_arrays[name])
        if values.shape[0] != selection.shape[0]:
            raise ValueError(
                f"Array '{name}' has {values.shape[0]} timesteps but mask has {selection.shape[0]}."
            )
        return values[selection]

    contact_state = selected("contact_state")
    force_history = selected("foot_contact_force_world_history")
    linear_velocity_world = selected("base_linear_velocity")
    angular_velocity_body = selected("base_angular_velocity_body")
    linear_acceleration_world = selected("base_linear_acceleration_world")
    angular_acceleration_body = selected("base_angular_acceleration_body")

    support_dynamics = compute_support_dynamics(contact_state, step_dt, foot_labels)
    impact_loading = compute_vertical_grf_dynamics(
        force_history,
        contact_state,
        robot_mass,
        foot_labels,
    )

    linear_velocity_summary = _summarize_vector_axes(linear_velocity_world)
    angular_velocity_summary = _summarize_vector_axes(angular_velocity_body)
    linear_acceleration_summary = _summarize_vector_axes(linear_acceleration_world)
    angular_acceleration_summary = _summarize_vector_axes(angular_acceleration_body)

    gravity = 9.81
    vertical_acceleration_g = linear_acceleration_world[:, 2] / gravity
    vertical_acceleration_summary = summarize_dynamics(vertical_acceleration_g)
    joint_demand = {
        "joint_acceleration": summarize_dynamics(selected("joint_accelerations")),
        "joint_velocity": summarize_dynamics(selected("joint_velocities")),
        "action_rate": summarize_dynamics(selected("action_rate")),
        "joint_torque": summarize_dynamics(selected("joint_torques")),
    }

    return {
        "analysis_sample_count": int(selection.sum()),
        "data_fidelity": dict(data_fidelity),
        "support_dynamics": support_dynamics,
        "base_excitation": {
            "vertical_velocity_rms_m_s": linear_velocity_summary["z"]["rms"],
            "vertical_velocity_mean_abs_m_s": linear_velocity_summary["z"]["mean_abs"],
            "vertical_acceleration_rms_g": vertical_acceleration_summary["rms"],
            "vertical_acceleration_mean_abs_g": vertical_acceleration_summary["mean_abs"],
            "vertical_acceleration_abs_p95_g": vertical_acceleration_summary["abs_p95"],
            "vertical_acceleration_abs_p99_g": vertical_acceleration_summary["abs_p99"],
            "pitch_rate_rms_rad_s": angular_velocity_summary["y"]["rms"],
            "pitch_rate_mean_abs_rad_s": angular_velocity_summary["y"]["mean_abs"],
            "pitch_acceleration_rms_rad_s2": angular_acceleration_summary["y"]["rms"],
            "pitch_acceleration_mean_abs_rad_s2": angular_acceleration_summary["y"]["mean_abs"],
            "linear_velocity_world": linear_velocity_summary,
            "angular_velocity_body": angular_velocity_summary,
            "linear_acceleration_world": linear_acceleration_summary,
            "angular_acceleration_body": angular_acceleration_summary,
        },
        "impact_loading": impact_loading,
        "joint_demand": joint_demand,
    }


def compute_swing_durations(contact_state: np.ndarray, sim_env_step_dt: float, foot_labels: list[str]) -> dict[str, list[float]]:
    """
    Returns raw swing durations (seconds) for every foot.
    contact_state : (T, F) 1 = contact, 0 = airborne
    """
    durations: dict[str, list[float]] = {lbl: [] for lbl in foot_labels}
    for foot_id, label in enumerate(foot_labels):
        in_contact = contact_state[:, foot_id].astype(bool)
        air_segments = compute_swing_segments(in_contact)
        # Convert segment lengths to seconds
        durations[label] = [(end_idx - start_idx) * sim_env_step_dt for start_idx, end_idx in air_segments]
    return durations


def compute_stance_durations(contact_state: np.ndarray, sim_env_step_dt: float, foot_labels: list[str]) -> dict[str, list[float]]:
    """
    Returns raw stance durations (seconds) for every foot.
    contact_state : (T, F) 1 = contact, 0 = airborne
    """
    durations: dict[str, list[float]] = {lbl: [] for lbl in foot_labels}
    for foot_id, label in enumerate(foot_labels):
        in_contact = contact_state[:, foot_id].astype(bool)
        contact_segments = compute_stance_segments(in_contact)
        # Convert segment lengths to seconds
        durations[label] = [(max(1, end_idx - 1) - start_idx) * sim_env_step_dt for start_idx, end_idx in contact_segments]
    return durations


def compute_swing_heights(contact_state: np.ndarray, foot_heights_contact: np.ndarray, reset_steps: list[int], foot_labels: list[str]) -> dict[str, list[float]]:
    """
    Returns max swing-phase height above terrain (contact frame) per foot.
    foot_heights_contact : (T, F) – Z in contact frame.
    """
    step_heights: dict[str, list[float]] = {lbl: [] for lbl in foot_labels}
    for fid, label in enumerate(foot_labels):
        in_contact = contact_state[:, fid].astype(bool)
        stance = compute_stance_segments(in_contact)
        for (s0, e0), (s1, _) in zip(stance, stance[1:]):
            if any(e0 <= r < s1 for r in reset_steps):
                continue
            if s1 - e0 > 0:
                h = np.nanmax(foot_heights_contact[e0:s1, fid])
                if not np.isnan(h):
                    step_heights[label].append(float(h))
    return step_heights


def compute_swing_lengths(contact_state: np.ndarray, foot_positions_world: np.ndarray, reset_steps: list[int], foot_labels: list[str]) -> dict[str, list[float]]:
    """
    Returns horizontal step length (world frame) per foot.
    """
    step_lengths: dict[str, list[float]] = {lbl: [] for lbl in foot_labels}
    for fid, label in enumerate(foot_labels):
        in_contact = contact_state[:, fid].astype(bool)
        stance = compute_stance_segments(in_contact)
        for (start_prev, _), (start_next, _) in zip(stance, stance[1:]):
            if any(start_prev < r <= start_next for r in reset_steps):
                continue
            d = np.linalg.norm(
                foot_positions_world[start_next, fid, :2] -
                foot_positions_world[start_prev, fid, :2]
            )
            if d < 1.5:          # filter teleports
                step_lengths[label].append(float(d))
    return step_lengths


def _constraint_normalization_scale(
    lower_bound: float | None,
    upper_bound: float | None,
) -> float | None:
    """Return the approved reference scale for cross-constraint excess percentages."""
    finite_lower = None if lower_bound is None else float(lower_bound)
    finite_upper = None if upper_bound is None else float(upper_bound)
    if finite_lower is not None and not np.isfinite(finite_lower):
        raise ValueError(f"lower_bound must be finite or None, got {lower_bound}.")
    if finite_upper is not None and not np.isfinite(finite_upper):
        raise ValueError(f"upper_bound must be finite or None, got {upper_bound}.")
    if finite_lower is None and finite_upper is None:
        return None

    if finite_lower is not None and finite_upper is not None:
        if finite_lower > finite_upper:
            raise ValueError(
                f"lower_bound must not exceed upper_bound, got {finite_lower} > {finite_upper}."
            )
        width = finite_upper - finite_lower
        if np.isclose(finite_lower, 0.0) or np.isclose(finite_upper, 0.0):
            scale = width
        elif width > 0.0:
            scale = width / 2.0
        else:
            scale = abs(finite_upper)
    else:
        finite_bound = finite_lower if finite_lower is not None else finite_upper
        scale = abs(float(finite_bound))

    return float(scale) if scale > 0.0 else None


def _summarize_constraint_violations(
    values: np.ndarray,
    lower_bound: float | None,
    upper_bound: float | None,
    element_labels: list[str],
    global_timestep_indices: np.ndarray,
) -> tuple[float | dict[str, float], dict[str, Any], dict[str, Any] | None]:
    """Summarize frequency and severity for one bounded constraint array."""
    array = np.asarray(values, dtype=np.float64)
    input_was_one_dimensional = array.ndim == 1
    if input_was_one_dimensional:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Constraint values must have shape (T,) or (T, E), got {array.shape}.")
    if array.shape[1] != len(element_labels):
        raise ValueError(
            f"Constraint values contain {array.shape[1]} elements but received "
            f"{len(element_labels)} labels."
        )
    timestep_indices = np.asarray(global_timestep_indices, dtype=np.int64)
    if timestep_indices.ndim != 1 or len(timestep_indices) != array.shape[0]:
        raise ValueError(
            "global_timestep_indices must match the constraint time dimension, got "
            f"{timestep_indices.shape} and {array.shape[0]}."
        )

    finite = np.isfinite(array)
    lower_excess = np.zeros_like(array)
    upper_excess = np.zeros_like(array)
    if lower_bound is not None:
        lower_excess = np.where(finite, np.maximum(float(lower_bound) - array, 0.0), 0.0)
    if upper_bound is not None:
        upper_excess = np.where(finite, np.maximum(array - float(upper_bound), 0.0), 0.0)
    absolute_excess = np.maximum(lower_excess, upper_excess)
    violation = finite & (absolute_excess > 0.0)
    normalization_scale = _constraint_normalization_scale(lower_bound, upper_bound)

    def severity_statistics(
        finite_mask: np.ndarray,
        violation_mask: np.ndarray,
        excess: np.ndarray,
    ) -> dict[str, float | None]:
        if not np.any(finite_mask):
            return {
                "mean_excess_when_violating": None,
                "max_excess": None,
                "mean_normalized_excess_percent_when_violating": None,
                "max_normalized_excess_percent": None,
            }
        violating_excess = excess[violation_mask]
        if violating_excess.size == 0:
            mean_excess = 0.0
            max_excess = 0.0
        else:
            mean_excess = float(violating_excess.mean())
            max_excess = float(violating_excess.max())
        if normalization_scale is None:
            mean_normalized = None
            max_normalized = None
        else:
            mean_normalized = float(100.0 * mean_excess / normalization_scale)
            max_normalized = float(100.0 * max_excess / normalization_scale)
        return {
            "mean_excess_when_violating": mean_excess,
            "max_excess": max_excess,
            "mean_normalized_excess_percent_when_violating": mean_normalized,
            "max_normalized_excess_percent": max_normalized,
        }

    percentages: dict[str, float] = {}
    per_element: dict[str, dict[str, float | None]] = {}
    for element_index, element_label in enumerate(element_labels):
        element_finite = finite[:, element_index]
        element_violation = violation[:, element_index]
        finite_count = int(element_finite.sum())
        percentages[element_label] = (
            100.0 * float(element_violation.sum()) / float(finite_count)
            if finite_count > 0
            else 0.0
        )
        per_element[element_label] = severity_statistics(
            element_finite,
            element_violation,
            absolute_excess[:, element_index],
        )

    magnitude_summary = {
        "lower_bound": None if lower_bound is None else float(lower_bound),
        "upper_bound": None if upper_bound is None else float(upper_bound),
        "normalization_scale": normalization_scale,
        "aggregate": severity_statistics(finite, violation, absolute_excess),
        "per_element": per_element,
    }

    maximum_violation = None
    if np.any(violation):
        flat_index = int(np.argmax(absolute_excess))
        timestep_index, element_index = np.unravel_index(flat_index, absolute_excess.shape)
        violates_lower = lower_excess[timestep_index, element_index] > 0.0
        violated_bound = lower_bound if violates_lower else upper_bound
        raw_excess = float(absolute_excess[timestep_index, element_index])
        maximum_violation = {
            "element": element_labels[element_index],
            "timestep": int(timestep_indices[timestep_index]),
            "analysis_sample_index": int(timestep_index),
            "observed_value": float(array[timestep_index, element_index]),
            "violated_bound": float(violated_bound),
            "direction": "lower" if violates_lower else "upper",
            "absolute_excess": raw_excess,
            "normalized_excess_percent": (
                None
                if normalization_scale is None
                else float(100.0 * raw_excess / normalization_scale)
            ),
        }

    frequency: float | dict[str, float]
    if input_was_one_dimensional:
        frequency = percentages[element_labels[0]]
    else:
        frequency = percentages
    return frequency, magnitude_summary, maximum_violation


def compute_summary_metrics(
    mask: np.ndarray,
    manual_reset_steps: List[int],
    automatic_reset_steps: List[int],
    data_arrays: Dict[str, np.ndarray],
    constants: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute summary metrics where mask is true.
    """
    step_dt = constants["step_dt"]
    joint_names = constants["joint_names"]
    foot_labels = constants["foot_labels"]
    constraint_bounds = constants["constraint_bounds"]
    total_robot_mass = constants["total_robot_mass"]

    mask_indices = np.where(mask)[0] # Get nonzero indices of mask, i.e. time steps that should be included in summary.
    global_to_local_mapping = {int(g): l for l, g in enumerate(mask_indices)} # Map global time step to local one.

    all_reset_steps = sorted(set(int(step) for step in manual_reset_steps + automatic_reset_steps))
    local_all_reset_steps = [global_to_local_mapping[step] for step in all_reset_steps if step in global_to_local_mapping]
    local_automatic_reset_steps = [global_to_local_mapping[int(step)] for step in automatic_reset_steps if int(step) in global_to_local_mapping]

    joint_positions = data_arrays["joint_positions"][mask]
    joint_velocities = data_arrays["joint_velocities"][mask]
    joint_torques = data_arrays["joint_torques"][mask]
    joint_accelerations = data_arrays["joint_accelerations"][mask]
    action_rates = data_arrays["action_rate"][mask]
    contact_force = data_arrays["contact_forces"][mask]
    base_position = data_arrays["base_position"][mask]
    base_orientation = data_arrays["base_orientation"][mask]
    base_linear_velocity = data_arrays["base_linear_velocity"][mask]
    base_angular_velocity = data_arrays["base_angular_velocity"][mask]
    base_linear_velocity_body = data_arrays["base_linear_velocity_body"][mask]
    base_angular_velocity_body = data_arrays["base_angular_velocity_body"][mask]
    commanded_velocity = data_arrays["commanded_velocity"][mask]
    contact_state = data_arrays["contact_state"][mask]
    foot_positions_world_frame = data_arrays["foot_positions_world_frame"][mask]
    foot_velocities_world_frame = data_arrays["foot_velocities_world_frame"][mask]
    foot_positions_contact_frame = data_arrays["foot_positions_contact_frame"][mask]
    distance_increment = data_arrays["distance_increment"][mask]
    reward = data_arrays["reward"][mask]
    power_array = data_arrays["power_array"][mask].copy()

    energy_per_joint = np.cumsum(np.abs(power_array), axis=0) * step_dt
    combined_energy = np.cumsum(np.abs(power_array).sum(axis=1)) * step_dt

    distance_walked_horizontal = float(distance_increment.sum())
    if distance_walked_horizontal > 1e-12:
        cost_of_transport = float(combined_energy[-1] / (total_robot_mass * 9.81 * distance_walked_horizontal))
    else:
        cost_of_transport = None

    # ---------- tracking / heading errors ---------------------------------
    linear_vel_x_error = commanded_velocity[:, 0] - base_linear_velocity_body[:, 0]
    linear_vel_y_error = commanded_velocity[:, 1] - base_linear_velocity_body[:, 1]
    yaw_error = commanded_velocity[:, 2] - base_angular_velocity_body[:, 2]
    linear_vel_x_rms = np.sqrt(np.mean(linear_vel_x_error**2))
    linear_vel_y_rms = np.sqrt(np.mean(linear_vel_y_error**2))
    yaw_rms = np.sqrt(np.mean(yaw_error**2))
    linear_vel_x_mean_abs = np.mean(np.abs(linear_vel_x_error))
    linear_vel_y_mean_abs = np.mean(np.abs(linear_vel_y_error))
    yaw_mean_abs = np.mean(np.abs(yaw_error))

    # ---------- constraint violations ------------------------------------
    violations = {}
    violation_magnitudes = {}
    maximum_constraint_violation = None
    constraint_metric_map = {
        "joint_velocity": (joint_velocities, joint_names),
        "joint_torque": (joint_torques, joint_names),
        "joint_acceleration": (joint_accelerations, joint_names),
        "action_rate": (action_rates, joint_names),
        "foot_contact_force": (contact_force, foot_labels),
        "joint_position": (joint_positions, joint_names),
        "air_time": ((1 - contact_state).astype(float), foot_labels),
    }
    joint_index_by_name = {joint_name: index for index, joint_name in enumerate(joint_names)}
    for term, (lb, ub) in constraint_bounds.items():
        metric_entry = constraint_metric_map.get(term)
        if metric_entry is None and term in joint_index_by_name:
            metric_entry = (joint_positions[:, joint_index_by_name[term]], [term])
        if metric_entry is None:
            continue
        metric_values, element_labels = metric_entry
        frequency, magnitude_summary, term_maximum = _summarize_constraint_violations(
            metric_values,
            lb,
            ub,
            list(element_labels),
            mask_indices,
        )
        violations[term] = frequency
        violation_magnitudes[term] = magnitude_summary
        if term_maximum is not None:
            term_maximum = {"constraint": term, **term_maximum}
            normalized_excess = term_maximum["normalized_excess_percent"]
            term_maximum["selection_basis"] = (
                "normalized_excess_percent"
                if normalized_excess is not None
                else "absolute_excess_fallback"
            )
            current_normalized_excess = (
                None
                if maximum_constraint_violation is None
                else maximum_constraint_violation["normalized_excess_percent"]
            )
            replace_maximum = maximum_constraint_violation is None
            if normalized_excess is not None:
                replace_maximum = (
                    current_normalized_excess is None
                    or normalized_excess > current_normalized_excess
                )
            elif current_normalized_excess is None and maximum_constraint_violation is not None:
                replace_maximum = (
                    term_maximum["absolute_excess"]
                    > maximum_constraint_violation["absolute_excess"]
                )
            if replace_maximum:
                maximum_constraint_violation = term_maximum

    # ---------- per-joint descriptive stats -------------
    summary_metric_map = {
        "position": joint_positions,
        "velocity": joint_velocities,
        "acceleration": joint_accelerations,
        "torque": joint_torques,
        "action_rate": action_rates,
        "energy": energy_per_joint,
        "power": power_array,
    }
    per_joint_summary = {}
    for joint_mapping, joint_name in enumerate(joint_names):
        per_joint_summary[joint_name] = {}
        for metric_name, data in summary_metric_map.items():
            col = data[:, joint_mapping]
            per_joint_summary[joint_name][metric_name] = summarize_metric(col.tolist())

    contact_force_summary = {}
    for foot_index, foot_label in enumerate(foot_labels):
        col = contact_force[:, foot_index]
        contact_force_summary[foot_label] = summarize_metric(col.tolist())

    swing_durations = compute_swing_durations(contact_state, step_dt, foot_labels)
    stance_durations = compute_stance_durations(contact_state, step_dt, foot_labels)
    step_height_data = compute_swing_heights(contact_state, foot_positions_contact_frame[:, :, 2], local_all_reset_steps, foot_labels)
    step_length_data = compute_swing_lengths(contact_state, foot_positions_world_frame, local_all_reset_steps, foot_labels)

    swing_duration_summary = {lbl: summarize_metric(data) for lbl, data in swing_durations.items()}
    stance_duration_summary = {lbl: summarize_metric(data) for lbl, data in stance_durations.items()}
    step_height_summary = {lbl: summarize_metric(data) for lbl, data in step_height_data.items()}
    step_length_summary = {lbl: summarize_metric(data) for lbl, data in step_length_data.items()}

    foot_velocity_world_frame_summary = compute_foot_velocity_summaries(contact_state, foot_velocities_world_frame, foot_labels)

    # ---------- symmetry TVD ---------------------------------------------
    joint_mapping = {jn: i for i, jn in enumerate(joint_names)}
    dofs = ["hip_joint", "thigh_joint", "calf_joint"]
    gait_symmetry_summary_per_dof = {d: {} for d in dofs}

    for dof in dofs:
        concatenated_joint_positions = np.concatenate([joint_positions[:, joint_mapping[f"{sr}_{dof}"]] for sr in ("FL", "FR", "RL", "RR")])
        optimal_num_bin_edges = optimal_bin_edges(concatenated_joint_positions, rule="fd")

        def pmf(sr):  # probability mass function
            return compute_histogram(joint_positions[:, joint_mapping[f"{sr}_{dof}"]], optimal_num_bin_edges)

        gait_symmetry_summary_per_dof[dof]["front_left_front_right"] = total_variation_distance(pmf("FL"), pmf("FR"))
        gait_symmetry_summary_per_dof[dof]["rear_left_rear_right"] = total_variation_distance(pmf("RL"), pmf("RR"))
        gait_symmetry_summary_per_dof[dof]["front_left_rear_left"] = total_variation_distance(pmf("FL"), pmf("RL"))
        gait_symmetry_summary_per_dof[dof]["front_right_rear_right"] = total_variation_distance(pmf("FR"), pmf("RR"))
        # New diagonal comparisons
        gait_symmetry_summary_per_dof[dof]["front_left_rear_right"] = total_variation_distance(pmf("FL"), pmf("RR"))
        gait_symmetry_summary_per_dof[dof]["front_right_rear_left"] = total_variation_distance(pmf("FR"), pmf("RL"))

    # Define all comparison keys, including the new ones
    comparison_keys = (
        "front_left_front_right", "rear_left_rear_right",
        "front_left_rear_left", "front_right_rear_right",
        "front_left_rear_right", "front_right_rear_left"
    )

    average_symmetry_tvd = {
        k: float(np.mean([gait_symmetry_summary_per_dof[d][k] for d in dofs]))
        for k in comparison_keys
    }

    axis_symmetry_tvd = {
        "left_vs_right": float(np.mean([average_symmetry_tvd["front_left_front_right"], average_symmetry_tvd["rear_left_rear_right"]])),
        "front_vs_rear": float(np.mean([average_symmetry_tvd["front_left_rear_left"], average_symmetry_tvd["front_right_rear_right"]])),
        # New diagonal axis summary
        "diagonal": float(np.mean([average_symmetry_tvd["front_left_rear_right"], average_symmetry_tvd["front_right_rear_left"]])),
    }

    return {
        "cumulative_unscaled_raw_reward": float(reward.sum()),
        "cumulative_reward_divided_by_cost_of_transport": (
            float(reward.sum() / cost_of_transport) if cost_of_transport is not None else None
        ),
        "cumulative_reward_divided_by_cost_of_transport_and_sim_time": (
            float(reward.sum() / (cost_of_transport * len(reward) * step_dt))
            if cost_of_transport is not None else None
        ),
        "base_linear_velocity_x_rms_error": float(linear_vel_x_rms),
        "base_linear_velocity_x_mean_abs_error": float(linear_vel_x_mean_abs),
        "base_linear_velocity_y_rms_error": float(linear_vel_y_rms),
        "base_linear_velocity_y_mean_abs_error": float(linear_vel_y_mean_abs),
        "base_angular_velocity_z_rms_error": float(yaw_rms),
        "base_angular_velocity_z_mean_abs_error": float(yaw_mean_abs),
        "per_joint_summary": per_joint_summary,
        "swing_duration_summary": swing_duration_summary,
        "stance_duration_summary": stance_duration_summary,
        "contact_force_summary": contact_force_summary,
        "step_length_summary": step_length_summary,
        "step_height_summary": step_height_summary,
        "foot_velocity_world_frame_summary": foot_velocity_world_frame_summary,
        "energy_consumption_per_joint": {jn: float(energy_per_joint[-1, j]) for j, jn in enumerate(joint_names)},
        "total_energy_consumption": float(combined_energy[-1]),
        "distance_walked_horizontal": distance_walked_horizontal,
        "cost_of_transport": cost_of_transport,
        "cumulative_power": summarize_metric(np.abs(power_array).sum(axis=1).tolist()),
        "constraint_violations_percent": violations,
        "constraint_violation_magnitudes": violation_magnitudes,
        "maximum_constraint_violation": maximum_constraint_violation,
        "gait_symmetry_tvd_by_joint": gait_symmetry_summary_per_dof,
        "aggregate_joint_symmetry_tvd": average_symmetry_tvd,
        "axis_symmetry_tvd": axis_symmetry_tvd,
    }


def compute_foot_velocity_summaries(contact_state: np.ndarray, foot_velocities: np.ndarray, foot_labels: list[str]) -> dict:
    """
    Computes summary statistics for foot velocities in the world frame, separated by stance and swing phase.
    """
    summaries = {}
    for foot_id, label in enumerate(foot_labels):
        in_contact = contact_state[:, foot_id].astype(bool)

        stance_velocities = foot_velocities[in_contact, foot_id, :]
        swing_velocities = foot_velocities[~in_contact, foot_id, :]

        summaries[label] = {
            "stance": {
                "x": summarize_metric(stance_velocities[:, 0].tolist()),
                "y": summarize_metric(stance_velocities[:, 1].tolist()),
                "z": summarize_metric(stance_velocities[:, 2].tolist()),
                "magnitude": summarize_metric(np.linalg.norm(stance_velocities, axis=1).tolist())
            },
            "swing": {
                "x": summarize_metric(swing_velocities[:, 0].tolist()),
                "y": summarize_metric(swing_velocities[:, 1].tolist()),
                "z": summarize_metric(swing_velocities[:, 2].tolist()),
                "magnitude": summarize_metric(np.linalg.norm(swing_velocities, axis=1).tolist())
            }
        }
    return summaries
