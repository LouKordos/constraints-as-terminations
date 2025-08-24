#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze a ROS 2 MCAP bag WITHOUT installing ROS (rosbags-only):
- Select a RELATIVE time window from the bag start for all parsing/analysis/plots
- Extract /joint_states (velocity, effort) -> per-joint power and total power/energy
- Compute Cost of Transport (CoT) given mass and linear velocity:
    * summary stats over window (net & positive-work)
    * CoT(t) curve over time within the window
- Plot IMU angular velocity (x,y,z) and orientation (quaternion -> Euler roll,pitch,yaw)
    * yaw is zeroed by subtracting the first yaw in the SELECTED window
- Plot Joint Torques (from /joint_states.effort) with ±20 Nm constraint lines
- Save tidy CSVs and PDF plots @ 600 dpi into sim2real_plots/

Dependencies:
  pip install rosbags pandas numpy matplotlib scipy
  (optional) pip install scienceplots   # for 'science','ieee' styles
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import math
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy: rotations + cumulative trapezoid for energy integration over time
try:
    from scipy.spatial.transform import Rotation as R
    from scipy.integrate import cumulative_trapezoid
except Exception:
    print("ERROR: SciPy is required. Install with: pip install scipy", file=sys.stderr)
    raise

# --------------------------------------------------------------------------------------
# Matplotlib style: emulate your script (scienceplots + rcParams) and apply globally
# --------------------------------------------------------------------------------------
def _apply_plot_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'ieee'])
    except Exception:
        pass

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 22,
        'figure.constrained_layout.use': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
    })

_apply_plot_style()

G_STANDARD = 9.80665  # m/s^2

def _as_list(field):
    """Robustly convert ROS array-like fields (possibly numpy arrays) to Python lists without boolean evaluation."""
    if field is None:
        return []
    if hasattr(field, "tolist"):
        try:
            return list(field.tolist())
        except Exception:
            pass
    try:
        return list(field)
    except TypeError:
        return []

def _plot_timeseries(
    df: pd.DataFrame,
    xcol: str,
    ycols: List[str],
    title: str,
    ylabel: str,
    outfile: Path,
    draw_hlines: Optional[List[Tuple[float, Dict]]] = None,
):
    """
    Style-consistent line plot helper for NON–joint-metric plots.
    Uses figsize (12,8) as requested.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    for col in ycols:
        if col in df.columns:
            ax.plot(df[xcol], df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    if draw_hlines:
        for yval, kw in draw_hlines:
            ax.axhline(yval, **kw)
    if len(ycols) > 1:
        ax.legend(loc='best')
    fig.tight_layout()
    outfile = outfile.with_suffix(".pdf")
    fig.savefig(outfile, dpi=600, format="pdf")
    plt.close(fig)

def _ensure_dir(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

@dataclass
class JointStateMsg:
    t_ns: int
    names: List[str]
    velocities: List[float]
    efforts: List[float]

@dataclass
class ImuMsg:
    t_ns: int
    gyro: Tuple[float, float, float]       # rad/s
    quat_xyzw: Tuple[float, float, float, float]  # (x,y,z,w)

class RosbagsReader:
    def __init__(self, bag_path: Path):
        try:
            from rosbags.highlevel import AnyReader
        except Exception as e:
            print("ERROR: rosbags is required. Install with: pip install rosbags", file=sys.stderr)
            raise
        self._AnyReader = AnyReader
        self._bag_path = bag_path
        self._reader = None

    def __enter__(self):
        self._reader = self._AnyReader([self._bag_path])
        self._reader.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._reader:
            self._reader.__exit__(exc_type, exc, tb)

    def list_topics(self) -> Dict[str, str]:
        return {c.topic: c.msgtype for c in self._reader.connections}

    def bag_start_time_ns(self) -> int:
        return int(self._reader.start_time)

    def iter_joint_states(self, topic: str, start_ns: Optional[int], stop_ns: Optional[int]) -> Iterable[JointStateMsg]:
        conns = [c for c in self._reader.connections if c.topic == topic]
        if not conns:
            return
        for c, t_ns, raw in self._reader.messages(connections=conns, start=start_ns, stop=stop_ns):
            msg = self._reader.deserialize(raw, c.msgtype)
            names = _as_list(getattr(msg, 'name', None))
            vel   = _as_list(getattr(msg, 'velocity', None))
            eff   = _as_list(getattr(msg, 'effort', None))
            yield JointStateMsg(int(t_ns), names, vel, eff)

    def iter_imu(self, topic: str, start_ns: Optional[int], stop_ns: Optional[int]) -> Iterable[ImuMsg]:
        conns = [c for c in self._reader.connections if c.topic == topic]
        if not conns:
            return
        for c, t_ns, raw in self._reader.messages(connections=conns, start=start_ns, stop=stop_ns):
            msg = self._reader.deserialize(raw, c.msgtype)
            gx = getattr(msg.angular_velocity, 'x', float('nan'))
            gy = getattr(msg.angular_velocity, 'y', float('nan'))
            gz = getattr(msg.angular_velocity, 'z', float('nan'))
            qx = getattr(msg.orientation, 'x', float('nan'))
            qy = getattr(msg.orientation, 'y', float('nan'))
            qz = getattr(msg.orientation, 'z', float('nan'))
            qw = getattr(msg.orientation, 'w', float('nan'))
            yield ImuMsg(int(t_ns), (gx, gy, gz), (qx, qy, qz, qw))

def compute_joint_power(js_iter: Iterable[JointStateMsg]) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    rows = []
    stats = dict(total_msgs=0, dropped_mismatch=0, dropped_nan=0)
    for msg in js_iter:
        stats['total_msgs'] += 1
        n = min(len(msg.names), len(msg.velocities), len(msg.efforts))
        if n == 0:
            continue
        if (len(msg.names) != len(msg.velocities)) or (len(msg.names) != len(msg.efforts)):
            stats['dropped_mismatch'] += (max(len(msg.names), len(msg.velocities), len(msg.efforts)) - n)
        for i in range(n):
            name = msg.names[i]
            v = msg.velocities[i]
            e = msg.efforts[i]
            if name is None or v is None or e is None or not np.isfinite([v, e]).all():
                stats['dropped_nan'] += 1
                continue
            rows.append({
                "t_ns": msg.t_ns,
                "t": pd.to_datetime(msg.t_ns, unit="ns"),
                "joint": name,
                "velocity": float(v),
                "effort": float(e),
                "power": float(e) * float(v),
            })

    if not rows:
        per_joint = pd.DataFrame(columns=["t_ns","t","joint","velocity","effort","power"])
        total_power = pd.DataFrame(columns=["t","total_power"])
        full_stats = {"total_msgs": stats['total_msgs'], **stats, "total_rows": 0, "unique_joints": 0}
        return per_joint, full_stats, total_power

    per_joint = pd.DataFrame(rows).sort_values(["t_ns","joint"]).reset_index(drop=True)
    total_power = (
        per_joint.groupby("t", as_index=False)[["power"]]
        .sum()
        .rename(columns={"power":"total_power"})
        .sort_values("t")
        .reset_index(drop=True)
    )
    full_stats = {
        "total_msgs": stats['total_msgs'],
        **stats,
        "total_rows": len(per_joint),
        "unique_joints": per_joint['joint'].nunique()
    }
    return per_joint, full_stats, total_power

def integrate_energy(total_power_df: pd.DataFrame) -> Dict[str, float]:
    if total_power_df.empty:
        return dict(duration_s=0.0, E_signed=0.0, E_positive=0.0, mean_power=0.0, mean_positive_power=0.0)

    t_ns = total_power_df["t"].astype("datetime64[ns]").astype("int64").to_numpy(dtype=np.int64)
    t_s = t_ns.astype(np.float64) * 1e-9
    P = total_power_df["total_power"].to_numpy(dtype=np.float64)

    if len(P) < 2:
        duration_s = 0.0
        return dict(duration_s=duration_s,
                    E_signed=0.0, E_positive=max(float(P[0]),0.0)*0.0,
                    mean_power=float(P[0]),
                    mean_positive_power=max(float(P[0]),0.0))

    E_signed = float(np.trapz(P, t_s))
    Ppos = np.maximum(P, 0.0)
    E_positive = float(np.trapz(Ppos, t_s))
    duration_s = float(t_s[-1] - t_s[0])
    mean_power = float(E_signed / duration_s) if duration_s > 0 else 0.0
    mean_positive_power = float(E_positive / duration_s) if duration_s > 0 else 0.0
    return dict(duration_s=duration_s, E_signed=E_signed, E_positive=E_positive,
                mean_power=mean_power, mean_positive_power=mean_positive_power)

def cot_timeseries(total_power_df: pd.DataFrame, mass_kg: Optional[float], speed_mps: Optional[float]) -> pd.DataFrame:
    res = pd.DataFrame(columns=["t","cot_net","cot_pos"])
    if total_power_df.empty or mass_kg is None or speed_mps is None or mass_kg <= 0 or speed_mps <= 0:
        return res

    t = total_power_df["t"].to_numpy()
    t_ns = total_power_df["t"].astype("datetime64[ns]").astype("int64").to_numpy(dtype=np.int64)
    t_s = t_ns.astype(np.float64) * 1e-9
    P = total_power_df["total_power"].to_numpy(dtype=np.float64)
    if len(P) < 2:
        return res

    E_net = cumulative_trapezoid(P, t_s, initial=0.0)
    Ppos = np.maximum(P, 0.0)
    E_pos = cumulative_trapezoid(Ppos, t_s, initial=0.0)

    t_rel = t_s - t_s[0]
    distance = speed_mps * t_rel
    denom = mass_kg * G_STANDARD * distance
    with np.errstate(divide='ignore', invalid='ignore'):
        cot_net = E_net / denom
        cot_pos = E_pos / denom
    cot_net[0] = np.nan
    cot_pos[0] = np.nan

    out = pd.DataFrame({"t": t, "cot_net": cot_net, "cot_pos": cot_pos})
    return out

def compute_cot_scalar(energy_j: float, mass_kg: float, speed_mps: float, duration_s: float) -> float:
    if mass_kg is None or speed_mps is None or duration_s <= 0 or mass_kg <= 0 or speed_mps <= 0:
        return float('nan')
    distance_m = speed_mps * duration_s
    denom = mass_kg * G_STANDARD * distance_m
    return float(energy_j / denom)

def imu_to_dfs(imu_iter: Iterable[ImuMsg], yaw_zero: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_g = []
    rows_q = []
    for m in imu_iter:
        rows_g.append({"t_ns": m.t_ns, "t": pd.to_datetime(m.t_ns, unit="ns"),
                       "gx": m.gyro[0], "gy": m.gyro[1], "gz": m.gyro[2]})
        rows_q.append({"t_ns": m.t_ns, "t": pd.to_datetime(m.t_ns, unit="ns"),
                       "qx": m.quat_xyzw[0], "qy": m.quat_xyzw[1], "qz": m.quat_xyzw[2], "qw": m.quat_xyzw[3]})

    if not rows_g:
        return pd.DataFrame(columns=["t","gx","gy","gz"]), pd.DataFrame(columns=["t","roll","pitch","yaw"])

    gyro_df = pd.DataFrame(rows_g).sort_values("t").reset_index(drop=True)
    qdf = pd.DataFrame(rows_q).sort_values("t").reset_index(drop=True)

    quats = qdf[["qx","qy","qz","qw"]].to_numpy(dtype=np.float64)
    valid = np.isfinite(quats).all(axis=1)
    euler = np.full((len(qdf), 3), np.nan, dtype=np.float64)
    if valid.any():
        r = R.from_quat(quats[valid])
        eul = r.as_euler('xyz', degrees=True)
        euler[valid] = eul
    euler_df = pd.DataFrame({"t": qdf["t"], "roll": euler[:,0], "pitch": euler[:,1], "yaw": euler[:,2]})

    # unwrap
    for col in ["roll","pitch","yaw"]:
        vals = euler_df[col].to_numpy()
        mask = np.isfinite(vals)
        vals_unwrap = np.unwrap(np.deg2rad(vals[mask]))
        euler_df.loc[mask, col] = np.rad2deg(vals_unwrap)

    if yaw_zero and not euler_df.empty:
        first_idx = euler_df["yaw"].first_valid_index()
        if first_idx is not None:
            yaw0 = float(euler_df.loc[first_idx, "yaw"])
            euler_df["yaw"] = euler_df["yaw"] - yaw0

    return gyro_df, euler_df

def autodetect_topics(topic_types: Dict[str, str], joint_hint: Optional[str], imu_hint: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    joint_topic = None
    imu_topic = None

    if joint_hint and joint_hint in topic_types:
        joint_topic = joint_hint
    if imu_hint and imu_hint in topic_types:
        imu_topic = imu_hint

    if joint_topic is None:
        for t, ty in topic_types.items():
            if ty and ("sensor_msgs/msg/JointState" in ty or ty.endswith("/JointState")):
                joint_topic = t
                break
        if joint_topic is None:
            for t in topic_types:
                if t.endswith("/joint_states") or t == "/joint_states":
                    joint_topic = t
                    break

    if imu_topic is None:
        for t, ty in topic_types.items():
            if ty and ("sensor_msgs/msg/Imu" in ty or ty.endswith("/Imu")):
                imu_topic = t
                break
        if imu_topic is None:
            for c in ["/imu", "/imu/data", "/imu0", "/filter/imu/data"]:
                if c in topic_types:
                    imu_topic = c
                    break

    return joint_topic, imu_topic

# ----------------------- Joint torque plotting (updated) -----------------------
def plot_joint_torques(per_joint_df: pd.DataFrame, outdir: Path, torque_limit_nm: float, t_start_ns: int):
    """
    Creates:
      - sim2real_plots/joint_metrics/torque/joint_torque_grid.pdf
      - sim2real_plots/joint_metrics/torque/joint_torque_overview.pdf
      - CSV: sim2real_plots/joint_metrics/torque/joint_torque_long.csv

    Updates:
      - Grid title: 'Joint Torques on Real Robot'
      - X-axis: relative time [s] from the selected window start
      - X-label: 'Time (s)'
    """
    if per_joint_df.empty or "effort" not in per_joint_df.columns:
        return

    torque_dir = outdir / "joint_metrics" / "torque"
    torque_dir.mkdir(parents=True, exist_ok=True)

    # Save long-form CSV (unchanged columns)
    per_joint_df[["t", "joint", "effort"]].to_csv(torque_dir / "joint_torque_long.csv", index=False)

    # Relative time [s] from specified analysis window start
    # Use the explicit window start (t_start_ns), not the first sample time.
    t_rel_s = (per_joint_df["t_ns"].to_numpy(dtype=np.int64) - int(t_start_ns)) * 1e-9
    per_joint_df = per_joint_df.assign(t_rel_s=t_rel_s)

    joints = sorted(per_joint_df["joint"].unique().tolist())
    n = len(joints)
    if n == 0:
        return

    # --- Grid: one subplot per joint (size unchanged; joint metrics are "fine") ---
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(6*ncols, 3.5*nrows))
    axes = np.atleast_2d(axes)
    line_kwargs = dict(linestyle='--', linewidth=1, color='red')

    for idx, joint in enumerate(joints):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        dfj = per_joint_df.loc[per_joint_df["joint"] == joint, ["t_rel_s", "effort"]]
        ax.plot(dfj["t_rel_s"], dfj["effort"], label=joint)
        ax.axhline(+torque_limit_nm, **line_kwargs, label='+20 Nm' if idx == 0 else None)
        ax.axhline(-torque_limit_nm, **line_kwargs, label='-20 Nm' if idx == 0 else None)
        ax.set_title(f"{joint}")
        ax.set_ylabel("Torque (N·m)")

        yabs = np.nanmax(np.abs(dfj["effort"].to_numpy())) if len(dfj) else torque_limit_nm
        ypad = max(0.05 * max(yabs, torque_limit_nm), 0.5)
        ymax = max(yabs, torque_limit_nm) + ypad
        ax.set_ylim(-ymax, +ymax)

        if idx == 0:
            ax.legend(loc='best')

    # Hide unused subplots
    for j in range(n, nrows*ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis('off')

    # X labels on bottom row
    for c in range(min(ncols, n)):
        axes[-1, c].set_xlabel("Time (s)")

    fig.suptitle("Joint Torques on Real Robot")
    fig.tight_layout()
    fig.savefig(_ensure_dir(torque_dir / "joint_torque_grid.pdf"), dpi=600, format="pdf")
    plt.close(fig)

    # --- Overview (keep size; switch to relative time for consistency) ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for joint in joints:
        dfj = per_joint_df.loc[per_joint_df["joint"] == joint, ["t_rel_s", "effort"]]
        ax2.plot(dfj["t_rel_s"], dfj["effort"], label=joint)
    ax2.axhline(+torque_limit_nm, **line_kwargs, label='+20 Nm limit')
    ax2.axhline(-torque_limit_nm, **line_kwargs, label='-20 Nm limit')
    ax2.set_title("Joint Torques — Overview")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Torque (N·m)")
    ax2.legend(ncol=2, loc='best')
    fig2.tight_layout()
    fig2.savefig(_ensure_dir(torque_dir / "joint_torque_overview.pdf"), dpi=600, format="pdf")
    plt.close(fig2)

# --------------------------------
# Main
# --------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Analyze ROS2 MCAP without ROS (rosbags-only): time-windowed joint power, CoT (scalar & over time), IMU plots, and joint torque plots with ±20 Nm limits."
    )
    ap.add_argument("--bag", required=True, help="Path to .mcap file")
    ap.add_argument("--joint-topic", default=None, help="Exact /joint_states topic (if None, autodetect)")
    ap.add_argument("--imu-topic", default=None, help="Exact IMU topic (if None, autodetect)")
    ap.add_argument("--mass-kg", type=float, default=None, help="System mass in kg (for CoT)")
    ap.add_argument("--linear-velocity", type=float, default=None, help="Forward speed in m/s (for CoT)")
    ap.add_argument("--t-start", type=float, default=0.0, help="Start time [s] RELATIVE TO BAG START for analysis/plots (default: 0.0)")
    ap.add_argument("--t-end", type=float, default=None, help="End time [s] RELATIVE TO BAG START (default: until bag end)")
    ap.add_argument("--outdir", default="sim2real_plots", help="Output directory for CSV + PDF plots (default: sim2real_plots)")
    args = ap.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"ERROR: Bag not found: {bag_path}", file=sys.stderr)
        sys.exit(2)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with RosbagsReader(bag_path) as r:
        topic_types = r.list_topics()
        joint_topic, imu_topic = autodetect_topics(topic_types, args.joint_topic, args.imu_topic)

        bag_start_ns = r.bag_start_time_ns()
        t_start_ns = bag_start_ns + int((args.t_start or 0.0) * 1e9)
        t_stop_ns = None if args.t_end is None else bag_start_ns + int(args.t_end * 1e9)
        if t_stop_ns is not None and t_stop_ns <= t_start_ns:
            print("ERROR: --t-end must be > --t-start", file=sys.stderr)
            sys.exit(3)

        print("\n=== Topic summary ===")
        print(f"Found {len(topic_types)} topics")
        print(f"JointState topic: {joint_topic if joint_topic else 'NOT FOUND'}")
        print(f"IMU topic: {imu_topic if imu_topic else 'NOT FOUND'}")
        print(f"Selected time window (relative to bag start): start={args.t_start:.3f}s  end={'end' if args.t_end is None else f'{args.t_end:.3f}s'}")

        # ---- Joint power & CoT ----
        per_joint_df = pd.DataFrame()
        total_power_df = pd.DataFrame()
        energy_info = dict(duration_s=0.0, E_signed=0.0, E_positive=0.0, mean_power=0.0, mean_positive_power=0.0)
        joint_stats = {}

        if joint_topic:
            per_joint_df, joint_stats, total_power_df = compute_joint_power(
                r.iter_joint_states(joint_topic, start_ns=t_start_ns, stop_ns=t_stop_ns)
            )

            if not per_joint_df.empty:
                per_joint_csv = outdir / "per_joint_power.csv"
                per_joint_df.to_csv(per_joint_csv, index=False)
                total_power_csv = outdir / "total_power.csv"
                total_power_df.to_csv(total_power_csv, index=False)

                energy_info = integrate_energy(total_power_df)

                if len(total_power_df) > 0:
                    _plot_timeseries(
                        total_power_df,
                        "t",
                        ["total_power"],
                        title="Total Mechanical Power",
                        ylabel="W",
                        outfile=_ensure_dir(outdir / "total_power"),
                    )

                # Joint torque plots (grid title + relative time + x label)
                plot_joint_torques(per_joint_df, outdir, torque_limit_nm=20.0, t_start_ns=t_start_ns)

        # ---- IMU plots ----
        gyro_df = pd.DataFrame()
        euler_df = pd.DataFrame()
        if imu_topic:
            gyro_df, euler_df = imu_to_dfs(r.iter_imu(imu_topic, start_ns=t_start_ns, stop_ns=t_stop_ns), yaw_zero=True)
            if not gyro_df.empty:
                gyro_csv = outdir / "imu_gyro.csv"
                gyro_df.to_csv(gyro_csv, index=False)
                _plot_timeseries(
                    gyro_df,
                    "t",
                    ["gx", "gy", "gz"],
                    title="IMU Angular Velocity",
                    ylabel="rad/s",
                    outfile=_ensure_dir(outdir / "imu_gyro"),
                )
            if not euler_df.empty:
                euler_csv = outdir / "imu_euler.csv"
                euler_df.to_csv(euler_csv, index=False)
                _plot_timeseries(
                    euler_df,
                    "t",
                    ["roll", "pitch", "yaw"],
                    title="IMU Orientation (Euler, xyz) — yaw zeroed",
                    ylabel="deg",
                    outfile=_ensure_dir(outdir / "imu_euler"),
                )

        # ---- CoT (scalar + time series) ----
        cot_net = float('nan')
        cot_pos = float('nan')
        cot_df = pd.DataFrame()
        if args.mass_kg is not None and args.linear_velocity is not None and energy_info["duration_s"] > 0.0:
            cot_net = compute_cot_scalar(energy_info["E_signed"], args.mass_kg, args.linear_velocity, energy_info["duration_s"])
            cot_pos = compute_cot_scalar(energy_info["E_positive"], args.mass_kg, args.linear_velocity, energy_info["duration_s"])
        if args.mass_kg is not None and args.linear_velocity is not None and not total_power_df.empty:
            cot_df = cot_timeseries(total_power_df, mass_kg=args.mass_kg, speed_mps=args.linear_velocity)
            if not cot_df.empty:
                cot_csv = outdir / "cot_timeseries.csv"
                cot_df.to_csv(cot_csv, index=False)
                _plot_timeseries(
                    cot_df,
                    "t",
                    ["cot_net", "cot_pos"],
                    title="Cost of Transport over time",
                    ylabel="dimensionless",
                    outfile=_ensure_dir(outdir / "cot_timeseries"),
                )

        # ---- Summary ----
        print("\n=== Analysis summary ===")
        print(f"Output directory: {outdir.resolve()}")
        if joint_topic and not per_joint_df.empty:
            print(f"[JointState] messages: {joint_stats.get('total_msgs',0)} | rows kept: {joint_stats.get('total_rows',0)} "
                  f"| joints: {joint_stats.get('unique_joints',0)} | dropped_mismatch: {joint_stats.get('dropped_mismatch',0)} | dropped_nan: {joint_stats.get('dropped_nan',0)}")
            print(f"Window duration: {energy_info['duration_s']:.3f} s")
            print(f"Energy (signed): {energy_info['E_signed']:.2f} J | Energy (positive): {energy_info['E_positive']:.2f} J")
            print(f"Mean Power (signed): {energy_info['mean_power']:.2f} W | Mean Power (positive): {energy_info['mean_positive_power']:.2f} W")
            if not math.isnan(cot_net):
                print(f"COT (net): {cot_net:.4f}  |  COT (positive-work): {cot_pos:.4f}")
            else:
                print("COT scalars: pass --mass-kg and --linear-velocity to compute.")
            print("Joint torque plots saved under: joint_metrics/torque/")
        else:
            print("JointState analysis skipped (topic not found or empty in window).")

        if imu_topic:
            print(f"[IMU] gyro samples: {len(gyro_df)} | euler samples: {len(euler_df)}")
            if not euler_df.empty:
                print("Yaw has been zeroed to the first yaw within the selected window.")
        else:
            print("IMU analysis skipped (topic not found).")

if __name__ == "__main__":
    main()
