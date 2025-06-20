from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

__all__ = [
    "compute_histogram",
    "compute_trimmed_histogram_data",
    "total_variation_distance",
    "optimal_bin_edges",
    "compute_stance_segments",
    "compute_energy_arrays",
    "summarize_metric",
    "compute_swing_durations",
    "compute_swing_heights",
    "compute_swing_lengths",
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
    
def compute_trimmed_histogram_data(data: np.ndarray, bins='auto', lower_percentile=0.5, upper_percentile=99.5):
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
        power_array[max(0, r - 1):r + 1, :]    = power_array[max(0, r - 1), :]

    energy_per_joint = np.cumsum(np.abs(power_array), axis=0) * step_dt
    combined_energy  = np.cumsum(np.abs(power_array).sum(axis=1)) * step_dt
    with np.errstate(divide="ignore", invalid="ignore"):
        cost_of_transport_time_series = np.abs(power_array).sum(axis=1) / (robot_mass * 9.81 * instantaneous_speed + 1e-12)
    return energy_per_joint, combined_energy, cost_of_transport_time_series

def summarize_metric(values: list[float]) -> dict[str, float]:
    """Return mean/min/max/… for a list, filling zeros if empty."""
    if not values:
        return {k: 0.0 for k in ("mean", "min", "max", "median", "90th_percentile", "99th_percentile", "stddev")}
    arr = np.asarray(values, dtype=float)
    return {
        "mean":             float(arr.mean()),
        "min":              float(arr.min()),
        "max":              float(arr.max()),
        "median":           float(np.median(arr)),
        "90th_percentile":  float(np.percentile(arr, 90)),
        "99th_percentile":  float(np.percentile(arr, 99)),
        "stddev":           float(arr.std()),
        "sum_signed_divided_by_num_steps":       float(np.sum(arr)) / float(len(values)),
        "sum_abs_divided_by_num_steps":          float(np.sum(np.abs(arr))) / float(len(values))
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
        durations[label] = [(max(1, end_idx-1) - start_idx) * sim_env_step_dt for start_idx, end_idx in contact_segments]
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

def compute_summary_metrics(mask: np.ndarray, reset_steps: List[int], data_arrays: Dict[str, np.ndarray], constants: Dict[str, Any],) -> Dict[str, Any]:
    """
    Compute summary metrics where mask is true
    """
    step_dt           = constants["step_dt"]
    joint_names       = constants["joint_names"]
    foot_labels       = constants["foot_labels"]
    constraint_bounds = constants["constraint_bounds"]
    total_robot_mass  = constants["total_robot_mass"]

    mask_indices = np.where(mask)[0] # Get nonzero indices of mask, i.e. time steps that should be included in summary. Same as mask.nonzero()
    global_to_local_mapping = {g: l for l, g in enumerate(mask_indices)} # Map global time step to local one, i.e. if we start summary at global index = 1000, local index should be 0 within this scope
    local_to_global_mapping = {l: g for l, g in enumerate(mask_indices)}

    joint_positions  = data_arrays["joint_positions"][mask]
    joint_velocities  = data_arrays["joint_velocities"][mask]
    joint_torques  = data_arrays["joint_torques"][mask]
    joint_accelerations  = data_arrays["joint_accelerations"][mask]
    action_rates  = data_arrays["action_rate"][mask]
    contact_force  = data_arrays["contact_forces"][mask]
    base_position  = data_arrays["base_position"][mask]
    base_orientation  = data_arrays["base_orientation"][mask]
    base_linear_velocity = data_arrays["base_linear_velocity"][mask]
    base_angular_velocity = data_arrays["base_angular_velocity"][mask]
    commanded_velocity  = data_arrays["commanded_velocity"][mask]
    contact_state  = data_arrays["contact_state"][mask]
    foot_positions_world_frame = data_arrays["foot_positions_world_frame"][mask]
    foot_positions_contact_frame = data_arrays["foot_positions_contact_frame"][mask]
    reward = data_arrays["reward"][mask]

    power_array = joint_torques * joint_velocities
    instantaneous_speed = np.linalg.norm(base_linear_velocity[:, :2], axis=1)
    # Fix transitions between resets, instantaneous speed and power might be invalid during those time steps
    for r in reset_steps:
        if r in global_to_local_mapping:
            local_index = global_to_local_mapping[r]
            instantaneous_speed[max(0, local_index - 1):local_index + 1] = instantaneous_speed[max(0, local_index - 1)]
            power_array[max(0, local_index - 1):local_index + 1, :] = power_array[max(0, local_index - 1), :]

    energy_per_joint = np.cumsum(np.abs(power_array), axis=0) * step_dt
    combined_energy  = np.cumsum(np.abs(power_array).sum(axis=1)) * step_dt

    with np.errstate(divide="ignore", invalid="ignore"):
        cost_of_transport_time_series  = np.abs(power_array).sum(axis=1) / (total_robot_mass * 9.81 * instantaneous_speed + 1e-12)
    mean_cost_of_transport = float(np.nanmean(cost_of_transport_time_series))

    # ---------- tracking / heading errors ---------------------------------
    linear_vel_x_rms = np.sqrt(np.mean((commanded_velocity[:, 0] - base_linear_velocity[:, 0])**2))
    linear_vel_y_rms = np.sqrt(np.mean((commanded_velocity[:, 1] - base_linear_velocity[:, 1])**2))
    yaw_rms    = np.sqrt(np.mean((commanded_velocity[:, 2] - base_angular_velocity[:, 2])**2))

    # ---------- constraint violations ------------------------------------
    violations = {}
    constraint_metric_map = {
        "joint_velocity"     : joint_velocities,
        "joint_torque"       : joint_torques,
        "joint_acceleration" : joint_accelerations,
        "action_rate"        : action_rates,
        "foot_contact_force" : contact_force.reshape(contact_force.shape[0], -1).mean(axis=1),
        "joint_position"     : joint_positions,
        "air_time"           : (1 - contact_state).astype(float),
    }
    for term, (lb, ub) in constraint_bounds.items():
        m = constraint_metric_map.get(term)
        if m is None:
            continue
        if m.ndim == 2: # per-joint term
            above = (ub is not None) & (m > ub)
            below = (lb is not None) & (m < lb)
            vmask = above | below
            violations[term] = dict(zip(joint_names, (vmask.mean(axis=0) * 100).tolist()))
        else: # global term
            above = (ub is not None) & (m > ub)
            below = (lb is not None) & (m < lb)
            vmask = above | below
            violations[term] = float(vmask.mean() * 100)

    # ---------- per-joint descriptive stats -------------
    summary_metric_map = {
        "position"    : joint_positions,
        "velocity"    : joint_velocities,
        "acceleration": joint_accelerations,
        "torque"      : joint_torques,
        "action_rate" : action_rates,
        "energy"      : energy_per_joint,
        "power"       : power_array,
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
    step_height_data = compute_swing_heights(contact_state, foot_positions_contact_frame[:, :, 2], reset_steps, foot_labels)
    step_length_data = compute_swing_lengths(contact_state, foot_positions_world_frame, reset_steps, foot_labels)

    swing_duration_summary = {lbl: summarize_metric(data) for lbl, data in swing_durations.items()}
    stance_duration_summary = {lbl: summarize_metric(data) for lbl, data in stance_durations.items()}
    step_height_summary = {lbl: summarize_metric(data) for lbl, data in step_height_data.items()}
    step_length_summary = {lbl: summarize_metric(data) for lbl, data in step_length_data.items()}

    # ---------- symmetry TVD ---------------------------------------------
    joint_mapping = {jn: i for i, jn in enumerate(joint_names)}
    dofs = ["hip_joint", "thigh_joint", "calf_joint"]
    gait_symmetry_summary_per_dof = {d: {} for d in dofs}

    for dof in dofs:
        concatenated_joint_positions = np.concatenate([joint_positions[:, joint_mapping[f"{sr}_{dof}"]] for sr in ("FL","FR","RL","RR")])
        optimal_num_bin_edges = optimal_bin_edges(concatenated_joint_positions, rule="fd")

        def pmf(sr):  # probability mass function
            return compute_histogram(joint_positions[:, joint_mapping[f"{sr}_{dof}"]], optimal_num_bin_edges)

        gait_symmetry_summary_per_dof[dof]["front_left_front_right"] = total_variation_distance(pmf("FL"), pmf("FR"))
        gait_symmetry_summary_per_dof[dof]["rear_left_rear_right"] = total_variation_distance(pmf("RL"), pmf("RR"))
        gait_symmetry_summary_per_dof[dof]["front_left_rear_left"] = total_variation_distance(pmf("FL"), pmf("RL"))
        gait_symmetry_summary_per_dof[dof]["front_right_rear_right"] = total_variation_distance(pmf("FR"), pmf("RR"))

    average_symmetry_tvd = {
        k: float(np.mean([gait_symmetry_summary_per_dof[d][k] for d in dofs]))
        for k in ("front_left_front_right","rear_left_rear_right", "front_left_rear_left","front_right_rear_right")
    }
    axis_symmetry_tvd = {
        "left_vs_right": float(np.mean([average_symmetry_tvd["front_left_front_right"], average_symmetry_tvd["rear_left_rear_right"]])),
        "front_vs_rear": float(np.mean([average_symmetry_tvd["front_left_rear_left"],   average_symmetry_tvd["front_right_rear_right"]])),
    }

    return {
        "cumulative_unscaled_raw_reward"                      : float(reward.sum()),
        "cumulative_reward_divided_by_cost_of_transport"      : float(reward.sum() / mean_cost_of_transport),
        "cumulative_reward_divided_by_cost_of_transport_and_sim_time": float(reward.sum() / (mean_cost_of_transport * len(reward) * step_dt)),
        "base_linear_velocity_x_rms_error"                    : float(linear_vel_x_rms),
        "base_linear_velocity_y_rms_error"                    : float(linear_vel_y_rms),
        "base_angular_velocity_z_rms_error"                   : float(yaw_rms),
        "per_joint_summary"                                   : per_joint_summary,
        "swing_duration_summary"                              : swing_duration_summary,
        "stance_duration_summary"                             : stance_duration_summary,
        "contact_force_summary"                               : contact_force_summary,
        "step_length_summary"                                 : step_length_summary,
        "step_height_summary"                                 : step_height_summary,
        "energy_consumption_per_joint"                        : {jn: float(energy_per_joint[-1, j]) for j, jn in enumerate(joint_names)},
        "total_energy_consumption"                            : float(combined_energy[-1]),
        "mean_cost_of_transport"                              : mean_cost_of_transport,
        "constraint_violations_percent"                       : violations,
        "gait_symmetry_tvd_by_joint"                          : gait_symmetry_summary_per_dof,
        "aggregate_joint_symmetry_tvd"                        : average_symmetry_tvd,
        "axis_symmetry_tvd"                                   : axis_symmetry_tvd,
    }