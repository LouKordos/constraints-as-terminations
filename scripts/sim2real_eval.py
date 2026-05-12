#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from rosbags.highlevel import AnyReader
except ImportError as exc:
    raise SystemExit(
        "Failed to import rosbags.highlevel.AnyReader. Install rosbags first, "
        "for example with `pip install rosbags`."
    ) from exc


@dataclass
class JointStateSeries:
    timestamps_ns: np.ndarray
    joint_names: List[str]
    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray


@dataclass
class BasePoseSeries:
    timestamps_ns: np.ndarray
    positions_xyz: np.ndarray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline sim2real evaluator for ROS 2 bags. "
            "It computes CoT from /joint_states effort*velocity and Vicon XY distance. "
            "The default reported CoT uses smoothed total absolute power and smoothed XY distance."
        )
    )
    parser.add_argument(
        "--bag",
        type=str,
        required=True,
        help="Path to bag directory or .mcap file.",
    )
    parser.add_argument(
        "--start-offset-sec",
        type=float,
        default=0.0,
        help="Relative start time from bag start in seconds.",
    )
    parser.add_argument(
        "--end-offset-sec",
        type=float,
        default=None,
        help=(
            "Relative end time from bag start in seconds. "
            "If it exceeds the bag or signal coverage, it is clamped automatically."
        ),
    )
    parser.add_argument(
        "--joint-state-topic",
        type=str,
        default="/joint_states",
        help="Topic providing sensor_msgs/msg/JointState.",
    )
    parser.add_argument(
        "--tf-topic",
        type=str,
        default="/tf",
        help="Topic providing tf2_msgs/msg/TFMessage.",
    )
    parser.add_argument(
        "--base-parent-frame",
        type=str,
        default="vicon/world",
        help="Parent/world frame for the base transform inside /tf.",
    )
    parser.add_argument(
        "--base-child-frame",
        type=str,
        default="vicon/Go2_Loukas/Go2_Loukas",
        help="Child/base frame for the robot transform inside /tf.",
    )
    parser.add_argument(
        "--robot-mass",
        type=float,
        default=20.0,
        help="Robot mass in kg used for CoT. Default: 20 kg.",
    )
    parser.add_argument(
        "--commanded-velocity",
        type=float,
        nargs=3,
        metavar=("LIN_X", "LIN_Y", "ANG_Z"),
        required=True,
        help=(
            "Constant commanded velocity for the selected bag slice: "
            "linear x, linear y, angular z. Stored in the summary for reference."
        ),
    )
    parser.add_argument(
        "--joint-velocity-limit",
        type=float,
        default=None,
        help="Absolute joint velocity limit for violation statistics.",
    )
    parser.add_argument(
        "--joint-torque-limit",
        type=float,
        default=None,
        help="Absolute joint torque/effort limit for violation statistics.",
    )
    parser.add_argument(
        "--joint-acceleration-limit",
        type=float,
        default=None,
        help="Absolute joint acceleration limit for violation statistics.",
    )
    parser.add_argument(
        "--max-allowed-gap-sec",
        type=float,
        default=0.25,
        help="Maximum allowed gap in required time series within the selected window before aborting.",
    )
    parser.add_argument(
        "--smoothing-window-sec",
        type=float,
        default=0.20,
        help=(
            "Centered moving-average window in seconds used for smoothing. "
            "The default reported CoT uses smoothed total absolute power and smoothed XY distance."
        ),
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=140,
        help="DPI used for saved plots.",
    )
    return parser.parse_args()


def normalize_frame_name(frame_name: str) -> str:
    return frame_name.lstrip("/")


def message_time_to_ns(msg: Any, bag_timestamp_ns: int) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return int(bag_timestamp_ns)
    return int(sec) * 1_000_000_000 + int(nanosec)


def tf_transform_time_to_ns(transform_stamped: Any, bag_timestamp_ns: int) -> int:
    header = getattr(transform_stamped, "header", None)
    stamp = getattr(header, "stamp", None)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return int(bag_timestamp_ns)
    return int(sec) * 1_000_000_000 + int(nanosec)


def summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": 0.0,
            "mean_of_abs": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "90th_percentile": 0.0,
            "99th_percentile": 0.0,
            "stddev": 0.0,
        }
    return {
        "mean": float(arr.mean()),
        "mean_of_abs": float(np.mean(np.abs(arr))),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "90th_percentile": float(np.percentile(arr, 90)),
        "99th_percentile": float(np.percentile(arr, 99)),
        "stddev": float(arr.std()),
    }


def reorder_joint_vector(
    names_in_message: Sequence[str],
    values: Sequence[float],
    canonical_names: Sequence[str],
) -> np.ndarray:
    mapped = np.full(len(canonical_names), np.nan, dtype=np.float64)
    source_lookup = {name: idx for idx, name in enumerate(names_in_message)}
    values_list = list(values)

    for idx, name in enumerate(canonical_names):
        source_idx = source_lookup.get(name)
        if source_idx is not None and source_idx < len(values_list):
            mapped[idx] = float(values_list[source_idx])

    return mapped


def apply_shared_sort_and_dedup(
    timestamps_ns: np.ndarray,
    *arrays: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    order = np.argsort(timestamps_ns)
    timestamps_sorted = timestamps_ns[order]
    arrays_sorted = [array[order] for array in arrays]

    unique_timestamps, unique_indices = np.unique(timestamps_sorted, return_index=True)
    arrays_unique = [array[unique_indices] for array in arrays_sorted]
    return unique_timestamps, arrays_unique


def collect_bag_data(
    args: argparse.Namespace,
) -> Tuple[int, int, JointStateSeries, BasePoseSeries, Dict[str, str]]:
    bag_path = Path(args.bag).resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    bag_paths = [bag_path]
    topic_types: Dict[str, str] = {}

    base_parent_frame = normalize_frame_name(args.base_parent_frame)
    base_child_frame = normalize_frame_name(args.base_child_frame)

    bag_start_ns: Optional[int] = None
    bag_end_ns: Optional[int] = None

    joint_state_timestamps: List[int] = []
    joint_state_positions: List[np.ndarray] = []
    joint_state_velocities: List[np.ndarray] = []
    joint_state_efforts: List[np.ndarray] = []
    canonical_joint_names: Optional[List[str]] = None

    base_pose_timestamps: List[int] = []
    base_positions_xyz: List[np.ndarray] = []

    with AnyReader(bag_paths) as reader:
        for connection in reader.connections:
            topic_types[connection.topic] = connection.msgtype

        for _, timestamp_ns, _ in reader.messages():
            timestamp_ns_int = int(timestamp_ns)
            if bag_start_ns is None:
                bag_start_ns = timestamp_ns_int
            bag_end_ns = timestamp_ns_int

        if bag_start_ns is None or bag_end_ns is None:
            raise RuntimeError(f"No messages were read from bag {bag_path}.")

        requested_topics = {args.joint_state_topic, args.tf_topic}
        selected_connections = [
            connection for connection in reader.connections if connection.topic in requested_topics
        ]

        for connection, bag_timestamp_ns, rawdata in reader.messages(connections=selected_connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.topic == args.joint_state_topic:
                msg_time_ns = message_time_to_ns(msg, int(bag_timestamp_ns))
                names = list(msg.name)

                if canonical_joint_names is None:
                    canonical_joint_names = names
                elif set(names) != set(canonical_joint_names):
                    raise RuntimeError(
                        "Joint name set in /joint_states changed across messages. "
                        "This script assumes a stable joint set."
                    )

                positions_field = getattr(msg, "position", None)
                velocities_field = getattr(msg, "velocity", None)
                efforts_field = getattr(msg, "effort", None)

                if positions_field is not None and len(positions_field) == len(names):
                    positions = reorder_joint_vector(names, positions_field, canonical_joint_names)
                else:
                    positions = np.full(len(canonical_joint_names), np.nan, dtype=np.float64)

                if velocities_field is not None and len(velocities_field) == len(names):
                    velocities = reorder_joint_vector(names, velocities_field, canonical_joint_names)
                else:
                    raise RuntimeError(
                        f"{args.joint_state_topic} does not contain usable velocity data. "
                        "Velocity is required for power and CoT."
                    )

                if efforts_field is not None and len(efforts_field) == len(names):
                    efforts = reorder_joint_vector(names, efforts_field, canonical_joint_names)
                else:
                    raise RuntimeError(
                        f"{args.joint_state_topic} does not contain usable effort data. "
                        "Effort is required for power and CoT."
                    )

                joint_state_timestamps.append(msg_time_ns)
                joint_state_positions.append(positions)
                joint_state_velocities.append(velocities)
                joint_state_efforts.append(efforts)

            elif connection.topic == args.tf_topic:
                transforms = getattr(msg, "transforms", [])
                for transform_stamped in transforms:
                    parent = normalize_frame_name(transform_stamped.header.frame_id)
                    child = normalize_frame_name(transform_stamped.child_frame_id)
                    if parent != base_parent_frame or child != base_child_frame:
                        continue

                    tf_time_ns = tf_transform_time_to_ns(transform_stamped, int(bag_timestamp_ns))
                    translation = transform_stamped.transform.translation

                    base_pose_timestamps.append(tf_time_ns)
                    base_positions_xyz.append(
                        np.array([translation.x, translation.y, translation.z], dtype=np.float64)
                    )

    if not joint_state_timestamps:
        raise RuntimeError(
            f"Required topic {args.joint_state_topic} was not found or contained no messages."
        )
    if not base_pose_timestamps:
        raise RuntimeError(
            f"Could not find the requested transform "
            f"{args.base_parent_frame} -> {args.base_child_frame} in {args.tf_topic}."
        )
    assert canonical_joint_names is not None

    joint_state_series = JointStateSeries(
        timestamps_ns=np.asarray(joint_state_timestamps, dtype=np.int64),
        joint_names=canonical_joint_names,
        positions=np.vstack(joint_state_positions),
        velocities=np.vstack(joint_state_velocities),
        efforts=np.vstack(joint_state_efforts),
    )
    base_pose_series = BasePoseSeries(
        timestamps_ns=np.asarray(base_pose_timestamps, dtype=np.int64),
        positions_xyz=np.vstack(base_positions_xyz),
    )

    joint_state_series.timestamps_ns, joint_arrays = apply_shared_sort_and_dedup(
        joint_state_series.timestamps_ns,
        joint_state_series.positions,
        joint_state_series.velocities,
        joint_state_series.efforts,
    )
    joint_state_series.positions, joint_state_series.velocities, joint_state_series.efforts = joint_arrays

    base_pose_series.timestamps_ns, base_arrays = apply_shared_sort_and_dedup(
        base_pose_series.timestamps_ns,
        base_pose_series.positions_xyz,
    )
    base_pose_series.positions_xyz = base_arrays[0]

    return int(bag_start_ns), int(bag_end_ns), joint_state_series, base_pose_series, topic_types


def check_max_gap(
    timestamps_ns: np.ndarray,
    name: str,
    selected_start_ns: int,
    selected_end_ns: int,
    max_allowed_gap_sec: float,
) -> None:
    in_window = timestamps_ns[
        (timestamps_ns >= selected_start_ns) & (timestamps_ns <= selected_end_ns)
    ]
    if in_window.size < 2:
        raise RuntimeError(f"Not enough {name} samples exist in the selected window.")
    max_gap_sec = float(np.max(np.diff(in_window.astype(np.float64))) * 1e-9)
    if max_gap_sec > max_allowed_gap_sec:
        raise RuntimeError(
            f"{name} has a maximum sample gap of {max_gap_sec:.6f}s in the selected window, "
            f"which exceeds --max-allowed-gap-sec={max_allowed_gap_sec:.6f}s."
        )


def build_output_dir(bag_path: Path, start_offset_sec: float, end_offset_sec: Optional[float]) -> Path:
    bag_container_dir = bag_path if bag_path.is_dir() else bag_path.parent

    def fmt(value: Optional[float]) -> str:
        if value is None:
            return "bag_end"
        return f"{value:.3f}".replace(".", "p").replace("-", "m")

    subdir_name = (
        f"sim2real_eval_start_{fmt(start_offset_sec)}"
        f"_end_{fmt(end_offset_sec)}"
    )
    return bag_container_dir / subdir_name


def interpolate_matrix(
    source_timestamps_ns: np.ndarray,
    source_values: np.ndarray,
    target_timestamps_ns: np.ndarray,
) -> np.ndarray:
    source_timestamps_sec = source_timestamps_ns.astype(np.float64) * 1e-9
    target_timestamps_sec = target_timestamps_ns.astype(np.float64) * 1e-9
    output = np.empty((target_timestamps_ns.shape[0], source_values.shape[1]), dtype=np.float64)

    for col in range(source_values.shape[1]):
        output[:, col] = np.interp(
            target_timestamps_sec,
            source_timestamps_sec,
            source_values[:, col],
        )

    return output


def window_size_in_samples(window_sec: float, timestamps_ns: np.ndarray) -> int:
    if timestamps_ns.size < 2 or window_sec <= 0.0:
        return 1
    dt_sec = np.diff(timestamps_ns.astype(np.float64)) * 1e-9
    positive_dt_sec = dt_sec[dt_sec > 0.0]
    if positive_dt_sec.size == 0:
        return 1
    median_dt_sec = float(np.median(positive_dt_sec))
    samples = max(1, int(round(window_sec / median_dt_sec)))
    if samples % 2 == 0:
        samples += 1
    if samples > timestamps_ns.size:
        samples = timestamps_ns.size if timestamps_ns.size % 2 == 1 else max(1, timestamps_ns.size - 1)
    return max(1, samples)


def centered_moving_average_1d(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values.astype(np.float64, copy=True)

    if values.size == 0:
        return values.astype(np.float64, copy=True)

    pad_left = window_size // 2
    pad_right = window_size // 2
    padded = np.pad(values.astype(np.float64), (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(padded, kernel, mode="valid")


def centered_moving_average_matrix(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values.astype(np.float64, copy=True)

    output = np.empty_like(values, dtype=np.float64)
    for col in range(values.shape[1]):
        output[:, col] = centered_moving_average_1d(values[:, col], window_size)
    return output


def cumulative_integral(values: np.ndarray, timestamps_ns: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("cumulative_integral expects a 1D array.")
    if values.shape[0] != timestamps_ns.shape[0]:
        raise ValueError("values and timestamps_ns must have the same length.")

    dt_sec = np.diff(timestamps_ns.astype(np.float64), prepend=timestamps_ns[0].astype(np.float64)) * 1e-9
    dt_sec[0] = 0.0
    return np.cumsum(values * dt_sec)


def compute_distance_increment_array(base_positions_xyz: np.ndarray) -> np.ndarray:
    distance_increment = np.zeros(base_positions_xyz.shape[0], dtype=np.float64)
    if base_positions_xyz.shape[0] > 1:
        distance_increment[1:] = np.linalg.norm(
            np.diff(base_positions_xyz[:, :2], axis=0),
            axis=1,
        )
    return distance_increment


def compute_global_violation_statistics(
    metric_array: np.ndarray,
    abs_limit: Optional[float],
) -> Optional[Dict[str, float]]:
    if abs_limit is None:
        return None

    finite_mask = np.isfinite(metric_array)
    if not np.any(finite_mask):
        return {
            "abs_limit": float(abs_limit),
            "percent_of_all_joint_samples": 0.0,
            "percent_of_timesteps_with_any_joint_violation": 0.0,
        }

    violation_mask = (np.abs(metric_array) > abs_limit) & finite_mask

    percent_of_all_joint_samples = 100.0 * float(violation_mask.sum()) / float(finite_mask.sum())

    valid_per_timestep = np.any(finite_mask, axis=1)
    any_violation_per_timestep = np.any(violation_mask, axis=1) & valid_per_timestep

    if np.any(valid_per_timestep):
        percent_of_timesteps_with_any_joint_violation = (
            100.0 * float(any_violation_per_timestep.sum()) / float(valid_per_timestep.sum())
        )
    else:
        percent_of_timesteps_with_any_joint_violation = 0.0

    return {
        "abs_limit": float(abs_limit),
        "percent_of_all_joint_samples": float(percent_of_all_joint_samples),
        "percent_of_timesteps_with_any_joint_violation": float(percent_of_timesteps_with_any_joint_violation),
    }


def safe_percentile_upper(values: np.ndarray, percentile: float = 99.5) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    upper = float(np.percentile(finite, percentile))
    if upper <= 0.0:
        upper = float(np.max(finite))
    if upper <= 0.0:
        return None
    return upper


def save_line_plot(
    output_path: Path,
    time_sec: np.ndarray,
    series: List[Tuple[np.ndarray, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    dpi: int,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(111)

    for y, label in series:
        ax.plot(time_sec, y, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_xy_trajectory_plot(
    output_path: Path,
    xy_raw: np.ndarray,
    xy_smoothed: np.ndarray,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(6.5, 6.0))
    ax = fig.add_subplot(111)

    ax.plot(xy_raw[:, 0], xy_raw[:, 1], label="raw XY trajectory")
    ax.plot(xy_smoothed[:, 0], xy_smoothed[:, 1], label="smoothed XY trajectory")

    ax.set_title("Base XY trajectory from Vicon")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def create_diagnostic_plots(
    *,
    plots_dir: Path,
    time_sec: np.ndarray,
    cumulative_distance_raw: np.ndarray,
    cumulative_distance_smoothed: np.ndarray,
    total_abs_power_raw: np.ndarray,
    total_abs_power_smoothed: np.ndarray,
    cumulative_energy_raw: np.ndarray,
    cumulative_energy_smoothed: np.ndarray,
    base_positions_xyz_raw: np.ndarray,
    base_positions_xyz_smoothed: np.ndarray,
    dpi: int,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_line_plot(
        output_path=plots_dir / "distance_over_time.png",
        time_sec=time_sec,
        series=[
            (cumulative_distance_raw, "raw cumulative XY distance"),
            (cumulative_distance_smoothed, "smoothed cumulative XY distance"),
        ],
        title="Horizontal distance over time",
        xlabel="time [s]",
        ylabel="distance [m]",
        dpi=dpi,
    )

    power_upper = safe_percentile_upper(total_abs_power_raw, percentile=99.5)
    power_ylim = None if power_upper is None else (0.0, power_upper * 1.15)
    save_line_plot(
        output_path=plots_dir / "total_absolute_power_over_time.png",
        time_sec=time_sec,
        series=[
            (total_abs_power_raw, "raw total absolute joint power"),
            (total_abs_power_smoothed, "smoothed total absolute joint power"),
        ],
        title="Total absolute joint power over time",
        xlabel="time [s]",
        ylabel="power [W]",
        dpi=dpi,
        ylim=power_ylim,
    )

    save_line_plot(
        output_path=plots_dir / "cumulative_energy_over_time.png",
        time_sec=time_sec,
        series=[
            (cumulative_energy_raw, "raw cumulative energy"),
            (cumulative_energy_smoothed, "smoothed cumulative energy"),
        ],
        title="Cumulative energy over time",
        xlabel="time [s]",
        ylabel="energy [J]",
        dpi=dpi,
    )

    save_xy_trajectory_plot(
        output_path=plots_dir / "base_xy_trajectory.png",
        xy_raw=base_positions_xyz_raw[:, :2],
        xy_smoothed=base_positions_xyz_smoothed[:, :2],
        dpi=dpi,
    )


def main() -> None:
    args = parse_arguments()

    bag_path = Path(args.bag).resolve()
    output_dir = build_output_dir(bag_path, args.start_offset_sec, args.end_offset_sec)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        bag_start_ns,
        bag_end_ns,
        joint_state_series,
        base_pose_series,
        topic_types,
    ) = collect_bag_data(args)

    usable_start_ns = max(
        int(bag_start_ns),
        int(joint_state_series.timestamps_ns[0]),
        int(base_pose_series.timestamps_ns[0]),
    )
    usable_end_ns = min(
        int(bag_end_ns),
        int(joint_state_series.timestamps_ns[-1]),
        int(base_pose_series.timestamps_ns[-1]),
    )

    requested_start_ns = bag_start_ns + int(round(args.start_offset_sec * 1e9))
    if args.end_offset_sec is None:
        requested_end_ns = usable_end_ns
    else:
        requested_end_ns = bag_start_ns + int(round(args.end_offset_sec * 1e9))

    selected_start_ns = max(usable_start_ns, requested_start_ns)
    selected_end_ns = min(usable_end_ns, requested_end_ns)

    if selected_end_ns <= selected_start_ns:
        raise ValueError(
            "Selected time window is empty after clamping to available data. "
            f"requested_start_ns={requested_start_ns}, requested_end_ns={requested_end_ns}, "
            f"usable_start_ns={usable_start_ns}, usable_end_ns={usable_end_ns}, "
            f"selected_start_ns={selected_start_ns}, selected_end_ns={selected_end_ns}"
        )

    check_max_gap(
        joint_state_series.timestamps_ns,
        "joint_states",
        selected_start_ns,
        selected_end_ns,
        args.max_allowed_gap_sec,
    )
    check_max_gap(
        base_pose_series.timestamps_ns,
        "base pose tf",
        selected_start_ns,
        selected_end_ns,
        args.max_allowed_gap_sec,
    )

    joint_mask = (
        (joint_state_series.timestamps_ns >= selected_start_ns)
        & (joint_state_series.timestamps_ns <= selected_end_ns)
    )
    eval_timestamps_ns = joint_state_series.timestamps_ns[joint_mask]
    if eval_timestamps_ns.size < 2:
        raise RuntimeError("Selected time window contains fewer than two /joint_states samples.")

    sim_times = (eval_timestamps_ns - eval_timestamps_ns[0]).astype(np.float64) * 1e-9
    dt_sec = np.diff(eval_timestamps_ns.astype(np.float64), prepend=eval_timestamps_ns[0].astype(np.float64)) * 1e-9
    dt_sec[0] = 0.0

    joint_positions = joint_state_series.positions[joint_mask]
    joint_velocities = joint_state_series.velocities[joint_mask]
    joint_torques = joint_state_series.efforts[joint_mask]

    if not np.isfinite(joint_velocities).all():
        raise RuntimeError("Selected joint velocities contain non-finite values.")
    if not np.isfinite(joint_torques).all():
        raise RuntimeError("Selected joint torques/efforts contain non-finite values.")

    base_positions_xyz_raw = interpolate_matrix(
        base_pose_series.timestamps_ns,
        base_pose_series.positions_xyz,
        eval_timestamps_ns,
    )

    smoothing_samples = window_size_in_samples(
        window_sec=args.smoothing_window_sec,
        timestamps_ns=eval_timestamps_ns,
    )

    base_positions_xyz_smoothed = centered_moving_average_matrix(
        base_positions_xyz_raw,
        smoothing_samples,
    )

    joint_accelerations = np.gradient(
        joint_velocities,
        eval_timestamps_ns.astype(np.float64) * 1e-9,
        axis=0,
    )

    joint_power_array = joint_torques * joint_velocities
    total_abs_power_raw = np.abs(joint_power_array).sum(axis=1)
    total_abs_power_smoothed = centered_moving_average_1d(
        total_abs_power_raw,
        smoothing_samples,
    )

    cumulative_energy_raw = cumulative_integral(
        total_abs_power_raw,
        eval_timestamps_ns,
    )
    cumulative_energy_smoothed = cumulative_integral(
        total_abs_power_smoothed,
        eval_timestamps_ns,
    )

    distance_increment_raw = compute_distance_increment_array(base_positions_xyz_raw)
    cumulative_distance_raw = np.cumsum(distance_increment_raw)
    distance_walked_horizontal_raw = float(cumulative_distance_raw[-1])

    distance_increment_smoothed = compute_distance_increment_array(base_positions_xyz_smoothed)
    cumulative_distance_smoothed = np.cumsum(distance_increment_smoothed)
    distance_walked_horizontal_smoothed = float(cumulative_distance_smoothed[-1])

    raw_total_energy_consumption = float(cumulative_energy_raw[-1])
    smoothed_total_energy_consumption = float(cumulative_energy_smoothed[-1])

    if distance_walked_horizontal_raw > 1e-12:
        raw_cost_of_transport = float(
            raw_total_energy_consumption / (args.robot_mass * 9.81 * distance_walked_horizontal_raw)
        )
    else:
        raw_cost_of_transport = None

    if distance_walked_horizontal_smoothed > 1e-12:
        smoothed_cost_of_transport = float(
            smoothed_total_energy_consumption / (args.robot_mass * 9.81 * distance_walked_horizontal_smoothed)
        )
    else:
        smoothed_cost_of_transport = None

    constraint_violations_percent = {
        "joint_velocity": compute_global_violation_statistics(
            joint_velocities,
            args.joint_velocity_limit,
        ),
        "joint_torque": compute_global_violation_statistics(
            joint_torques,
            args.joint_torque_limit,
        ),
        "joint_acceleration": compute_global_violation_statistics(
            joint_accelerations,
            args.joint_acceleration_limit,
        ),
    }

    median_joint_state_dt_sec = float(np.median(np.diff(eval_timestamps_ns.astype(np.float64)) * 1e-9))
    estimated_joint_state_rate_hz = float(1.0 / median_joint_state_dt_sec) if median_joint_state_dt_sec > 0.0 else None

    summary_metrics: Dict[str, Any] = {
        "bag_path": str(bag_path),
        "output_dir": str(output_dir),
        "bag_start_time_ns": int(bag_start_ns),
        "bag_end_time_ns": int(bag_end_ns),
        "usable_start_time_ns": int(usable_start_ns),
        "usable_end_time_ns": int(usable_end_ns),
        "requested_start_offset_sec": float(args.start_offset_sec),
        "requested_end_offset_sec": float(args.end_offset_sec) if args.end_offset_sec is not None else None,
        "requested_start_time_ns": int(requested_start_ns),
        "requested_end_time_ns": int(requested_end_ns),
        "selected_start_time_ns": int(selected_start_ns),
        "selected_end_time_ns": int(selected_end_ns),
        "selected_start_offset_sec": float((selected_start_ns - bag_start_ns) * 1e-9),
        "selected_end_offset_sec": float((selected_end_ns - bag_start_ns) * 1e-9),
        "selected_duration_sec": float((selected_end_ns - selected_start_ns) * 1e-9),
        "start_was_clamped": bool(selected_start_ns != requested_start_ns),
        "end_was_clamped": bool(selected_end_ns != requested_end_ns),
        "num_eval_samples": int(eval_timestamps_ns.shape[0]),
        "median_joint_state_dt_sec": median_joint_state_dt_sec,
        "estimated_joint_state_rate_hz": estimated_joint_state_rate_hz,
        "joint_names": list(joint_state_series.joint_names),
        "robot_mass": float(args.robot_mass),
        "commanded_velocity": {
            "linear_x": float(args.commanded_velocity[0]),
            "linear_y": float(args.commanded_velocity[1]),
            "angular_z": float(args.commanded_velocity[2]),
        },
        "reported_metric_variant": "smoothed",
        "power_source": "joint_states.effort * joint_states.velocity",
        "power_smoothing": {
            "method": "centered_moving_average",
            "window_sec": float(args.smoothing_window_sec),
            "window_samples": int(smoothing_samples),
            "applied_to": "total absolute joint power time series",
        },
        "distance_source": "Vicon tf XY position increments",
        "distance_smoothing": {
            "method": "centered_moving_average",
            "window_sec": float(args.smoothing_window_sec),
            "window_samples": int(smoothing_samples),
            "applied_to": "interpolated base XYZ position before XY increment accumulation",
        },
        "cost_of_transport": smoothed_cost_of_transport,
        "raw_cost_of_transport": raw_cost_of_transport,
        "total_energy_consumption": smoothed_total_energy_consumption,
        "raw_total_energy_consumption": raw_total_energy_consumption,
        "distance_walked_horizontal": float(distance_walked_horizontal_smoothed),
        "raw_distance_walked_horizontal": float(distance_walked_horizontal_raw),
        "total_absolute_power_raw_summary": summarize_metric(total_abs_power_raw.tolist()),
        "total_absolute_power_smoothed_summary": summarize_metric(total_abs_power_smoothed.tolist()),
        "constraint_violations_percent": constraint_violations_percent,
        "topic_types": topic_types,
        "topics_used": {
            "joint_state_topic": args.joint_state_topic,
            "tf_topic": args.tf_topic,
        },
        "frames_used": {
            "base_parent_frame": normalize_frame_name(args.base_parent_frame),
            "base_child_frame": normalize_frame_name(args.base_child_frame),
        },
        "notes": [
            "The default reported CoT uses smoothed total absolute power and smoothed XY distance.",
            "Raw CoT, raw energy, and raw horizontal distance are also saved for comparison.",
            "Horizontal distance is accumulated in the XY plane using Euclidean increments, not only x or only y.",
            "Constraint violations are reported as aggregate percentages across all joint-time samples and as percentage of timesteps with any violating joint.",
            "The evaluation timeline is the selected /joint_states timestamps. Vicon positions are interpolated onto that timeline.",
            "If the requested time window exceeds the available bag or signal coverage, it is clamped automatically.",
        ],
    }

    npz_path = output_dir / "sim2real_eval_data.npz"
    np.savez(
        npz_path,
        eval_timestamps_ns=eval_timestamps_ns,
        sim_times=sim_times,
        dt_sec=dt_sec,
        joint_names=np.asarray(joint_state_series.joint_names, dtype=object),
        joint_positions_array=joint_positions,
        joint_velocities_array=joint_velocities,
        joint_torques_array=joint_torques,
        joint_accelerations_array=joint_accelerations,
        joint_power_array=joint_power_array,
        total_absolute_power_raw=total_abs_power_raw,
        total_absolute_power_smoothed=total_abs_power_smoothed,
        cumulative_energy_raw=cumulative_energy_raw,
        cumulative_energy_smoothed=cumulative_energy_smoothed,
        base_position_array=base_positions_xyz_raw,
        base_position_smoothed_array=base_positions_xyz_smoothed,
        distance_increment_array_raw=distance_increment_raw,
        cumulative_distance_array_raw=cumulative_distance_raw,
        distance_increment_array_smoothed=distance_increment_smoothed,
        cumulative_distance_array_smoothed=cumulative_distance_smoothed,
        total_robot_mass=float(args.robot_mass),
        smoothing_window_sec=float(args.smoothing_window_sec),
        smoothing_window_samples=int(smoothing_samples),
        requested_start_ns=int(requested_start_ns),
        requested_end_ns=int(requested_end_ns),
        selected_start_ns=int(selected_start_ns),
        selected_end_ns=int(selected_end_ns),
        usable_start_ns=int(usable_start_ns),
        usable_end_ns=int(usable_end_ns),
    )

    plots_dir = output_dir / "plots"
    create_diagnostic_plots(
        plots_dir=plots_dir,
        time_sec=sim_times,
        cumulative_distance_raw=cumulative_distance_raw,
        cumulative_distance_smoothed=cumulative_distance_smoothed,
        total_abs_power_raw=total_abs_power_raw,
        total_abs_power_smoothed=total_abs_power_smoothed,
        cumulative_energy_raw=cumulative_energy_raw,
        cumulative_energy_smoothed=cumulative_energy_smoothed,
        base_positions_xyz_raw=base_positions_xyz_raw,
        base_positions_xyz_smoothed=base_positions_xyz_smoothed,
        dpi=args.plot_dpi,
    )

    summary_path = output_dir / "sim2real_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary_metrics, file, indent=4)

    print(f"Wrote summary metrics to {summary_path}")
    print(f"Wrote aligned evaluation arrays to {npz_path}")
    print(f"Wrote diagnostic plots to {plots_dir}")


if __name__ == "__main__":
    main()