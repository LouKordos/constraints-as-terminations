#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rosbags.highlevel import AnyReader
except ImportError as exc:
    raise SystemExit(
        "Failed to import rosbags.highlevel.AnyReader. Install rosbags first, "
        "for example with `pip install rosbags`."
    ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline sim2real evaluator for ROS 2 bags. "
            "It computes CoT using the same core formula as the simulation evaluator: "
            "integrated absolute joint power divided by m*g*horizontal distance. "
            "Joint power is computed from /joint_states effort * velocity."
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
        help="Relative end time from bag start in seconds. Defaults to the latest usable time.",
    )
    parser.add_argument(
        "--resample-dt",
        type=float,
        default=None,
        help="Optional fixed resample dt in seconds. Defaults to the median /joint_states dt in the selected window.",
    )
    parser.add_argument(
        "--joint-state-topic",
        type=str,
        default="/joint_states",
        help="Topic providing sensor_msgs/msg/JointState.",
    )
    parser.add_argument(
        "--imu-topic",
        type=str,
        default="/imu",
        help="Topic providing sensor_msgs/msg/Imu. Optional. Used for body angular velocity if available.",
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
            "Constant commanded body velocity for the selected bag slice: "
            "linear x, linear y, angular z."
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
            "Centered moving-average window in seconds used for diagnostic plot overlays "
            "and smoothed-distance diagnostics. The main reported CoT still uses raw power "
            "and raw horizontal distance."
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
    quaternions_xyzw: np.ndarray


@dataclass
class ImuSeries:
    timestamps_ns: np.ndarray
    angular_velocity_xyz: np.ndarray


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
) -> Tuple[int, int, JointStateSeries, BasePoseSeries, Optional[ImuSeries], Dict[str, str]]:
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
    base_quaternions_xyzw: List[np.ndarray] = []

    imu_timestamps: List[int] = []
    imu_angular_velocity_xyz: List[np.ndarray] = []

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

        requested_topics = {args.joint_state_topic, args.tf_topic, args.imu_topic}
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
                    rotation = transform_stamped.transform.rotation

                    base_pose_timestamps.append(tf_time_ns)
                    base_positions_xyz.append(
                        np.array([translation.x, translation.y, translation.z], dtype=np.float64)
                    )
                    base_quaternions_xyzw.append(
                        np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64)
                    )

            elif connection.topic == args.imu_topic:
                msg_time_ns = message_time_to_ns(msg, int(bag_timestamp_ns))
                angular_velocity = msg.angular_velocity
                imu_timestamps.append(msg_time_ns)
                imu_angular_velocity_xyz.append(
                    np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z], dtype=np.float64)
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
        quaternions_xyzw=np.vstack(base_quaternions_xyzw),
    )

    imu_series = None
    if imu_timestamps:
        imu_series = ImuSeries(
            timestamps_ns=np.asarray(imu_timestamps, dtype=np.int64),
            angular_velocity_xyz=np.vstack(imu_angular_velocity_xyz),
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
        base_pose_series.quaternions_xyzw,
    )
    base_pose_series.positions_xyz, base_pose_series.quaternions_xyzw = base_arrays

    if imu_series is not None:
        imu_series.timestamps_ns, imu_arrays = apply_shared_sort_and_dedup(
            imu_series.timestamps_ns,
            imu_series.angular_velocity_xyz,
        )
        imu_series.angular_velocity_xyz = imu_arrays[0]

    return int(bag_start_ns), int(bag_end_ns), joint_state_series, base_pose_series, imu_series, topic_types


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


def infer_reference_dt(
    selected_start_ns: int,
    selected_end_ns: int,
    joint_state_series: JointStateSeries,
    override_dt: Optional[float],
) -> float:
    if override_dt is not None:
        if override_dt <= 0.0:
            raise ValueError("--resample-dt must be positive.")
        return float(override_dt)

    in_window = joint_state_series.timestamps_ns[
        (joint_state_series.timestamps_ns >= selected_start_ns)
        & (joint_state_series.timestamps_ns <= selected_end_ns)
    ]
    if in_window.size < 2:
        raise RuntimeError("Not enough /joint_states samples in the selected window to infer dt.")

    diffs = np.diff(in_window.astype(np.float64)) * 1e-9
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        raise RuntimeError("Could not infer a positive resample dt from /joint_states timestamps.")
    return float(np.median(positive))


def build_uniform_time_axis(start_ns: int, end_ns: int, dt_sec: float) -> np.ndarray:
    step_ns = int(round(dt_sec * 1e9))
    num_steps = int(math.floor((end_ns - start_ns) / step_ns)) + 1
    if num_steps < 2:
        raise RuntimeError("Selected time window is too short after applying dt.")
    return start_ns + np.arange(num_steps, dtype=np.int64) * step_ns


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


def nearest_neighbor_matrix(
    source_timestamps_ns: np.ndarray,
    source_values: np.ndarray,
    target_timestamps_ns: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(source_timestamps_ns, target_timestamps_ns, side="left")
    idx = np.clip(idx, 0, len(source_timestamps_ns) - 1)

    prev_idx = np.clip(idx - 1, 0, len(source_timestamps_ns) - 1)
    use_prev = np.abs(source_timestamps_ns[prev_idx] - target_timestamps_ns) <= np.abs(
        source_timestamps_ns[idx] - target_timestamps_ns
    )
    final_idx = np.where(use_prev, prev_idx, idx)
    return source_values[final_idx]


def quaternion_conjugate_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    output = quat_xyzw.copy()
    output[..., :3] *= -1.0
    return output


def quaternion_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.stack((x, y, z, w), axis=-1)


def rotate_vectors_by_quaternion_xyzw(vectors_xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    zeros = np.zeros((vectors_xyz.shape[0], 1), dtype=np.float64)
    vec_quat = np.concatenate((vectors_xyz, zeros), axis=1)
    quat_conj = quaternion_conjugate_xyzw(quat_xyzw)
    return quaternion_multiply_xyzw(
        quaternion_multiply_xyzw(quat_xyzw, vec_quat),
        quat_conj,
    )[:, :3]


def world_to_body_vectors(vectors_world_xyz: np.ndarray, body_quat_xyzw: np.ndarray) -> np.ndarray:
    return rotate_vectors_by_quaternion_xyzw(vectors_world_xyz, quaternion_conjugate_xyzw(body_quat_xyzw))


def quaternion_to_euler_zyx_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    x = quat_xyzw[:, 0]
    y = quat_xyzw[:, 1]
    z = quat_xyzw[:, 2]
    w = quat_xyzw[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (np.pi / 2.0), np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack((yaw, pitch, roll), axis=1)


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


def moving_average_1d(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values.copy()
    if window_size > values.shape[0]:
        window_size = values.shape[0] if values.shape[0] % 2 == 1 else max(1, values.shape[0] - 1)
    if window_size <= 1:
        return values.copy()

    pad_left = window_size // 2
    pad_right = window_size // 2
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(padded, kernel, mode="valid")


def moving_average_matrix(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values.copy()
    output = np.empty_like(values, dtype=np.float64)
    for col in range(values.shape[1]):
        output[:, col] = moving_average_1d(values[:, col], window_size)
    return output


def smoothing_window_samples(window_sec: float, step_dt: float, num_steps: int) -> int:
    if window_sec <= 0.0:
        return 1
    samples = max(1, int(round(window_sec / step_dt)))
    if samples % 2 == 0:
        samples += 1
    if samples > num_steps:
        samples = num_steps if num_steps % 2 == 1 else max(1, num_steps - 1)
    return max(1, samples)


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
    horizontal_speed_raw: np.ndarray,
    horizontal_speed_smoothed: np.ndarray,
    instantaneous_cot_raw: np.ndarray,
    instantaneous_cot_smoothed: np.ndarray,
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
            (cumulative_distance_smoothed, "smoothed-position cumulative XY distance"),
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
        series=[(cumulative_energy_raw, "cumulative absolute joint energy")],
        title="Cumulative energy over time",
        xlabel="time [s]",
        ylabel="energy [J]",
        dpi=dpi,
    )

    speed_upper = safe_percentile_upper(horizontal_speed_raw, percentile=99.5)
    speed_ylim = None if speed_upper is None else (0.0, speed_upper * 1.15)
    save_line_plot(
        output_path=plots_dir / "horizontal_speed_over_time.png",
        time_sec=time_sec,
        series=[
            (horizontal_speed_raw, "raw horizontal speed from Vicon"),
            (horizontal_speed_smoothed, "smoothed horizontal speed from Vicon"),
        ],
        title="Horizontal speed over time",
        xlabel="time [s]",
        ylabel="speed [m/s]",
        dpi=dpi,
        ylim=speed_ylim,
    )

    cot_upper = safe_percentile_upper(instantaneous_cot_raw, percentile=99.0)
    cot_ylim = None if cot_upper is None else (0.0, cot_upper * 1.15)
    save_line_plot(
        output_path=plots_dir / "instantaneous_cot_diagnostic_over_time.png",
        time_sec=time_sec,
        series=[
            (instantaneous_cot_raw, "raw instantaneous CoT diagnostic"),
            (instantaneous_cot_smoothed, "smoothed instantaneous CoT diagnostic"),
        ],
        title="Instantaneous CoT diagnostic over time",
        xlabel="time [s]",
        ylabel="CoT [-]",
        dpi=dpi,
        ylim=cot_ylim,
    )

    save_xy_trajectory_plot(
        output_path=plots_dir / "base_xy_trajectory.png",
        xy_raw=base_positions_xyz_raw[:, :2],
        xy_smoothed=base_positions_xyz_smoothed[:, :2],
        dpi=dpi,
    )


def compute_summary(
    *,
    joint_names: Sequence[str],
    step_dt: float,
    robot_mass: float,
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    joint_torques: np.ndarray,
    joint_accelerations: np.ndarray,
    power_array: np.ndarray,
    base_positions_xyz_raw: np.ndarray,
    base_positions_xyz_smoothed: np.ndarray,
    base_linear_velocity_world_xyz_for_summary: np.ndarray,
    base_linear_velocity_body_xyz_for_summary: np.ndarray,
    base_angular_velocity_body_xyz: np.ndarray,
    commanded_velocity_xyz: np.ndarray,
    args: argparse.Namespace,
    topic_types: Dict[str, str],
    bag_start_ns: int,
    bag_end_ns: int,
    requested_start_ns: int,
    requested_end_ns: int,
    selected_start_ns: int,
    selected_end_ns: int,
    usable_start_ns: int,
    usable_end_ns: int,
    smoothing_window_samples_used: int,
) -> Dict[str, Any]:
    energy_per_joint = np.cumsum(np.abs(power_array), axis=0) * step_dt
    combined_energy = np.cumsum(np.abs(power_array).sum(axis=1)) * step_dt

    distance_increment_raw = compute_distance_increment_array(base_positions_xyz_raw)
    distance_walked_horizontal_raw = float(distance_increment_raw.sum())

    distance_increment_smoothed = compute_distance_increment_array(base_positions_xyz_smoothed)
    distance_walked_horizontal_smoothed = float(distance_increment_smoothed.sum())

    if distance_walked_horizontal_raw > 1e-12:
        cost_of_transport = float(
            combined_energy[-1] / (robot_mass * 9.81 * distance_walked_horizontal_raw)
        )
    else:
        cost_of_transport = None

    if distance_walked_horizontal_smoothed > 1e-12:
        diagnostic_cost_of_transport_smoothed_xy_distance = float(
            combined_energy[-1] / (robot_mass * 9.81 * distance_walked_horizontal_smoothed)
        )
    else:
        diagnostic_cost_of_transport_smoothed_xy_distance = None

    base_speed_horizontal_for_summary = np.linalg.norm(base_linear_velocity_world_xyz_for_summary[:, :2], axis=1)

    linear_vel_x_rms = np.sqrt(
        np.mean((commanded_velocity_xyz[:, 0] - base_linear_velocity_body_xyz_for_summary[:, 0]) ** 2)
    )
    linear_vel_y_rms = np.sqrt(
        np.mean((commanded_velocity_xyz[:, 1] - base_linear_velocity_body_xyz_for_summary[:, 1]) ** 2)
    )
    yaw_rms = np.sqrt(
        np.mean((commanded_velocity_xyz[:, 2] - base_angular_velocity_body_xyz[:, 2]) ** 2)
    )

    per_joint_summary: Dict[str, Any] = {}
    summary_metric_map = {
        "position": joint_positions,
        "velocity": joint_velocities,
        "acceleration": joint_accelerations,
        "torque": joint_torques,
        "energy": energy_per_joint,
        "power": power_array,
    }

    for joint_idx, joint_name in enumerate(joint_names):
        per_joint_summary[joint_name] = {}
        for metric_name, metric_values in summary_metric_map.items():
            per_joint_summary[joint_name][metric_name] = summarize_metric(
                metric_values[:, joint_idx].tolist()
            )

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

    actual_start_offset_sec = float((selected_start_ns - bag_start_ns) * 1e-9)
    actual_end_offset_sec = float((selected_end_ns - bag_start_ns) * 1e-9)

    summary: Dict[str, Any] = {
        "bag_path": str(Path(args.bag).resolve()),
        "output_dir": str(build_output_dir(Path(args.bag).resolve(), args.start_offset_sec, args.end_offset_sec)),
        "bag_start_time_ns": int(bag_start_ns),
        "bag_end_time_ns": int(bag_end_ns),
        "usable_start_time_ns": int(usable_start_ns),
        "usable_end_time_ns": int(usable_end_ns),
        "requested_start_offset_sec": float(args.start_offset_sec),
        "requested_end_offset_sec": float(args.end_offset_sec) if args.end_offset_sec is not None else None,
        "requested_start_time_ns": int(requested_start_ns),
        "requested_end_time_ns": int(requested_end_ns),
        "selected_start_offset_sec": actual_start_offset_sec,
        "selected_end_offset_sec": actual_end_offset_sec,
        "selected_start_time_ns": int(selected_start_ns),
        "selected_end_time_ns": int(selected_end_ns),
        "selected_duration_sec": float((selected_end_ns - selected_start_ns) * 1e-9),
        "start_was_clamped": bool(selected_start_ns != requested_start_ns),
        "end_was_clamped": bool(selected_end_ns != requested_end_ns),
        "step_dt": float(step_dt),
        "num_eval_steps": int(joint_positions.shape[0]),
        "joint_names": list(joint_names),
        "robot_mass": float(robot_mass),
        "power_source": "joint_states.effort * joint_states.velocity",
        "commanded_velocity": {
            "linear_x": float(args.commanded_velocity[0]),
            "linear_y": float(args.commanded_velocity[1]),
            "angular_z": float(args.commanded_velocity[2]),
        },
        "distance_walked_horizontal": float(distance_walked_horizontal_raw),
        "diagnostic_distance_walked_horizontal_smoothed_xy": float(distance_walked_horizontal_smoothed),
        "distance_computation": "sum over selected window of Euclidean XY position increments from Vicon tf",
        "cost_of_transport": cost_of_transport,
        "diagnostic_cost_of_transport_smoothed_xy_distance": diagnostic_cost_of_transport_smoothed_xy_distance,
        "total_energy_consumption": float(combined_energy[-1]),
        "energy_consumption_per_joint": {
            joint_name: float(energy_per_joint[-1, joint_idx])
            for joint_idx, joint_name in enumerate(joint_names)
        },
        "cumulative_power": summarize_metric(np.abs(power_array).sum(axis=1).tolist()),
        "mean_base_speed_horizontal": float(base_speed_horizontal_for_summary.mean()),
        "base_speed_horizontal_summary": summarize_metric(base_speed_horizontal_for_summary.tolist()),
        "base_linear_velocity_body_x_summary": summarize_metric(base_linear_velocity_body_xyz_for_summary[:, 0].tolist()),
        "base_linear_velocity_body_y_summary": summarize_metric(base_linear_velocity_body_xyz_for_summary[:, 1].tolist()),
        "base_angular_velocity_body_z_summary": summarize_metric(base_angular_velocity_body_xyz[:, 2].tolist()),
        "base_linear_velocity_x_rms_error": float(linear_vel_x_rms),
        "base_linear_velocity_y_rms_error": float(linear_vel_y_rms),
        "base_angular_velocity_z_rms_error": float(yaw_rms),
        "constraint_violations_percent": constraint_violations_percent,
        "per_joint_summary": per_joint_summary,
        "topic_types": topic_types,
        "topics_used": {
            "joint_state_topic": args.joint_state_topic,
            "imu_topic": args.imu_topic,
            "tf_topic": args.tf_topic,
        },
        "frames_used": {
            "base_parent_frame": normalize_frame_name(args.base_parent_frame),
            "base_child_frame": normalize_frame_name(args.base_child_frame),
        },
        "diagnostic_smoothing": {
            "smoothing_window_sec": float(args.smoothing_window_sec),
            "smoothing_window_samples": int(smoothing_window_samples_used),
            "note": (
                "Smoothing is used for diagnostic overlays and smoothed-distance sensitivity analysis. "
                "The main reported cost_of_transport still uses raw power and raw horizontal distance."
            ),
        },
        "notes": [
            "CoT uses the same core formula as the existing sim evaluator: sum_t sum_j |power_j(t)| * dt / (mass * g * horizontal_distance).",
            "Here power_j(t) is computed from /joint_states effort * velocity.",
            "Horizontal distance is accumulated in the XY plane using Euclidean position increments, not only x or only y.",
            "Constraint violations are reported as aggregate percentages across all joint-time samples, plus percentage of timesteps with any violating joint.",
            "No lowstate-dependent metrics or gait/contact diagram metrics are computed.",
            "Body linear velocity summaries are based on a smoothed Vicon position derivative to reduce differentiation noise.",
            "If the requested time window exceeds the available range of the required signals, it is clamped automatically.",
        ],
    }

    return summary


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
        imu_series,
        topic_types,
    ) = collect_bag_data(args)

    usable_start_candidates = [
        int(bag_start_ns),
        int(joint_state_series.timestamps_ns[0]),
        int(base_pose_series.timestamps_ns[0]),
    ]
    usable_end_candidates = [
        int(bag_end_ns),
        int(joint_state_series.timestamps_ns[-1]),
        int(base_pose_series.timestamps_ns[-1]),
    ]

    if imu_series is not None:
        usable_start_candidates.append(int(imu_series.timestamps_ns[0]))
        usable_end_candidates.append(int(imu_series.timestamps_ns[-1]))

    usable_start_ns = max(usable_start_candidates)
    usable_end_ns = min(usable_end_candidates)

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

    if imu_series is not None:
        check_max_gap(
            imu_series.timestamps_ns,
            "imu",
            selected_start_ns,
            selected_end_ns,
            args.max_allowed_gap_sec,
        )

    step_dt = infer_reference_dt(
        selected_start_ns=selected_start_ns,
        selected_end_ns=selected_end_ns,
        joint_state_series=joint_state_series,
        override_dt=args.resample_dt,
    )

    eval_timestamps_ns = build_uniform_time_axis(
        selected_start_ns,
        selected_end_ns,
        step_dt,
    )
    sim_times = (eval_timestamps_ns - eval_timestamps_ns[0]).astype(np.float64) * 1e-9

    joint_positions = interpolate_matrix(
        joint_state_series.timestamps_ns,
        joint_state_series.positions,
        eval_timestamps_ns,
    )
    joint_velocities = interpolate_matrix(
        joint_state_series.timestamps_ns,
        joint_state_series.velocities,
        eval_timestamps_ns,
    )
    joint_torques = interpolate_matrix(
        joint_state_series.timestamps_ns,
        joint_state_series.efforts,
        eval_timestamps_ns,
    )

    if not np.isfinite(joint_velocities).all():
        raise RuntimeError("Interpolated joint velocities contain non-finite values.")
    if not np.isfinite(joint_torques).all():
        raise RuntimeError("Interpolated joint torques/efforts contain non-finite values.")

    power_array = joint_torques * joint_velocities
    joint_accelerations = np.gradient(joint_velocities, step_dt, axis=0)

    base_positions_xyz_raw = interpolate_matrix(
        base_pose_series.timestamps_ns,
        base_pose_series.positions_xyz,
        eval_timestamps_ns,
    )
    base_quaternions_xyzw = nearest_neighbor_matrix(
        base_pose_series.timestamps_ns,
        base_pose_series.quaternions_xyzw,
        eval_timestamps_ns,
    )
    quat_norms = np.linalg.norm(base_quaternions_xyzw, axis=1, keepdims=True)
    base_quaternions_xyzw = base_quaternions_xyzw / np.maximum(quat_norms, 1e-12)

    smoothing_samples = smoothing_window_samples(
        window_sec=args.smoothing_window_sec,
        step_dt=step_dt,
        num_steps=eval_timestamps_ns.shape[0],
    )

    base_positions_xyz_smoothed = moving_average_matrix(base_positions_xyz_raw, smoothing_samples)

    base_linear_velocity_world_xyz_raw = np.gradient(base_positions_xyz_raw, step_dt, axis=0)
    base_linear_velocity_world_xyz_smoothed = np.gradient(base_positions_xyz_smoothed, step_dt, axis=0)

    base_linear_velocity_body_xyz_raw = world_to_body_vectors(
        base_linear_velocity_world_xyz_raw,
        base_quaternions_xyzw,
    )
    base_linear_velocity_body_xyz_smoothed = world_to_body_vectors(
        base_linear_velocity_world_xyz_smoothed,
        base_quaternions_xyzw,
    )

    if imu_series is not None:
        base_angular_velocity_body_xyz = interpolate_matrix(
            imu_series.timestamps_ns,
            imu_series.angular_velocity_xyz,
            eval_timestamps_ns,
        )
    else:
        # Fallback: estimate yaw rate from Vicon orientation.
        yaw_pitch_roll = np.unwrap(
            quaternion_to_euler_zyx_xyzw(base_quaternions_xyzw),
            axis=0,
        )
        yaw_rate = np.gradient(yaw_pitch_roll[:, 0], step_dt)
        base_angular_velocity_body_xyz = np.column_stack(
            [
                np.zeros_like(yaw_rate),
                np.zeros_like(yaw_rate),
                yaw_rate,
            ]
        )

    commanded_velocity_xyz = np.tile(
        np.asarray(args.commanded_velocity, dtype=np.float64).reshape(1, 3),
        (eval_timestamps_ns.shape[0], 1),
    )

    total_abs_power_raw = np.abs(power_array).sum(axis=1)
    total_abs_power_smoothed = moving_average_1d(total_abs_power_raw, smoothing_samples)

    cumulative_energy_raw = np.cumsum(total_abs_power_raw) * step_dt

    distance_increment_raw = compute_distance_increment_array(base_positions_xyz_raw)
    cumulative_distance_raw = np.cumsum(distance_increment_raw)

    distance_increment_smoothed = compute_distance_increment_array(base_positions_xyz_smoothed)
    cumulative_distance_smoothed = np.cumsum(distance_increment_smoothed)

    horizontal_speed_raw = np.linalg.norm(base_linear_velocity_world_xyz_raw[:, :2], axis=1)
    horizontal_speed_smoothed = np.linalg.norm(base_linear_velocity_world_xyz_smoothed[:, :2], axis=1)

    instantaneous_cot_raw = total_abs_power_raw / (args.robot_mass * 9.81 * horizontal_speed_raw + 1e-12)
    instantaneous_cot_smoothed = total_abs_power_smoothed / (args.robot_mass * 9.81 * horizontal_speed_smoothed + 1e-12)

    summary_metrics = compute_summary(
        joint_names=joint_state_series.joint_names,
        step_dt=step_dt,
        robot_mass=args.robot_mass,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        joint_torques=joint_torques,
        joint_accelerations=joint_accelerations,
        power_array=power_array,
        base_positions_xyz_raw=base_positions_xyz_raw,
        base_positions_xyz_smoothed=base_positions_xyz_smoothed,
        base_linear_velocity_world_xyz_for_summary=base_linear_velocity_world_xyz_smoothed,
        base_linear_velocity_body_xyz_for_summary=base_linear_velocity_body_xyz_smoothed,
        base_angular_velocity_body_xyz=base_angular_velocity_body_xyz,
        commanded_velocity_xyz=commanded_velocity_xyz,
        args=args,
        topic_types=topic_types,
        bag_start_ns=bag_start_ns,
        bag_end_ns=bag_end_ns,
        requested_start_ns=requested_start_ns,
        requested_end_ns=requested_end_ns,
        selected_start_ns=selected_start_ns,
        selected_end_ns=selected_end_ns,
        usable_start_ns=usable_start_ns,
        usable_end_ns=usable_end_ns,
        smoothing_window_samples_used=smoothing_samples,
    )

    npz_path = output_dir / "sim2real_eval_data.npz"
    np.savez(
        npz_path,
        eval_timestamps_ns=eval_timestamps_ns,
        sim_times=sim_times,
        joint_positions_array=joint_positions,
        joint_velocities_array=joint_velocities,
        joint_torques_array=joint_torques,
        joint_accelerations_array=joint_accelerations,
        power_array=power_array,
        total_absolute_power_raw=total_abs_power_raw,
        total_absolute_power_smoothed=total_abs_power_smoothed,
        cumulative_energy_raw=cumulative_energy_raw,
        base_position_array=base_positions_xyz_raw,
        base_position_smoothed_array=base_positions_xyz_smoothed,
        base_linear_velocity_array_raw=base_linear_velocity_world_xyz_raw,
        base_linear_velocity_array_smoothed=base_linear_velocity_world_xyz_smoothed,
        base_linear_velocity_body_array_raw=base_linear_velocity_body_xyz_raw,
        base_linear_velocity_body_array_smoothed=base_linear_velocity_body_xyz_smoothed,
        base_angular_velocity_body_array=base_angular_velocity_body_xyz,
        commanded_velocity_array=commanded_velocity_xyz,
        distance_increment_array_raw=distance_increment_raw,
        cumulative_distance_array_raw=cumulative_distance_raw,
        distance_increment_array_smoothed=distance_increment_smoothed,
        cumulative_distance_array_smoothed=cumulative_distance_smoothed,
        horizontal_speed_raw=horizontal_speed_raw,
        horizontal_speed_smoothed=horizontal_speed_smoothed,
        instantaneous_cot_raw=instantaneous_cot_raw,
        instantaneous_cot_smoothed=instantaneous_cot_smoothed,
        joint_names=np.asarray(joint_state_series.joint_names, dtype=object),
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
        horizontal_speed_raw=horizontal_speed_raw,
        horizontal_speed_smoothed=horizontal_speed_smoothed,
        instantaneous_cot_raw=instantaneous_cot_raw,
        instantaneous_cot_smoothed=instantaneous_cot_smoothed,
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