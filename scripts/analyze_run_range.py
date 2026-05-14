import argparse
import json
import logging
import platform
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401
except ImportError:
    scienceplots = None

import wandb

from metrics_utils import compute_swing_heights, summarize_metric


DEFAULT_WANDB_METRICS = [
    "Curriculum/terrain_levels",
    "Episode_Reward/track_lin_vel_xy_exp",
]

DEFAULT_SUMMARY_WANDB_METRICS = [
    "Curriculum/terrain_levels",
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode/MaxJointPos",
    "Episode/MaxAirTime",
    "Episode/EnergyConsumed",
]

DEFAULT_LABEL_MAP = {
    "Curriculum/terrain_levels": "Terrain level",
    "Episode_Reward/track_lin_vel_xy_exp": "Velocity tracking reward",
    "Episode/MaxJointPos": "Max. joint position (rad)",
    "Episode/MaxAirTime": "Max. air time (s)",
    "Episode/EnergyConsumed": "Energy consumed (J)",
}

DEFAULT_SUMMARY_LABEL_MAP = {
    "Curriculum/terrain_levels": "Final Terrain Level",
    "Episode_Reward/track_lin_vel_xy_exp": "Final Velocity Tracking Reward",
    "Episode/MaxJointPos": "Final Max Joint Position",
    "Episode/MaxAirTime": "Final Max Air Time",
    "Episode/EnergyConsumed": "Final Energy Consumed",
    "mean_cost_of_transport_range": "Mean Cost of Transport (velocity range)",
    "violation_torque": "Max Constraint Violation (Torque %)",
    "violation_accel": "Max Constraint Violation (Acceleration %)",
    "rms_error_x": "Base Velocity RMS Error (X)",
    "rms_error_y": "Base Velocity RMS Error (Y)",
    "rms_error_xy_mean": "Mean Base Velocity RMS Error (X,Y)",
}

DEFAULT_STEP_HEIGHT_FLAT_SCENARIO_TAG = "walk_x_flat_terrain_1.0mps"
DEFAULT_STEP_HEIGHT_UNEVEN_SCENARIO_TAG = "medium_walk_x_uneven_terrain"
ALL_TIME_PLACEHOLDER = "ALL"

# A long, publication-oriented qualitative palette. The first eight colors are based on
# the Okabe-Ito colorblind-safe palette; the remaining colors extend the cycle for plots
# with many baselines. Repeated colors are further disambiguated by line style and marker.
PAPER_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#332288",  # indigo
    "#AA4499",  # purple
    "#DDCC77",  # sand
    "#88CCEE",  # cyan
    "#117733",  # dark green
]
PAPER_LINESTYLES = ["-", "--", "-.", ":"]
PAPER_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "h", "*", "p"]
BASELINE_COLOR = "0.25"


@dataclass
class SeriesSpec:
    label: str
    env_name: str
    wandb_path: str
    start_dt: datetime | None
    end_dt: datetime | None


@dataclass
class JsonRunData:
    env_name: str
    run_name: str
    checkpoint: int | None
    json_path: Path
    cot_df: pd.DataFrame | None
    summary: dict[str, Any]
    metrics_summary: dict[str, Any]


@dataclass
class SeriesData:
    label: str
    env_name: str
    wandb_path: str
    wandb_runs: dict[str, dict[str, Any]]
    json_runs: dict[str, JsonRunData]
    selected_wandb_run_names: list[str]
    selected_json_run_names: list[str]


def sanitize_filename(name: str) -> str:
    return re.sub(r'[/\\?%*:|"<> ]+', "_", name).strip("_")


def parse_run_name_to_datetime(run_name: str) -> datetime | None:
    try:
        return datetime.strptime(run_name, "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        return None


def parse_flexible_datetime_or_all(dt_str: str, is_end_date: bool = False) -> datetime | None:
    if dt_str.upper() == ALL_TIME_PLACEHOLDER:
        return None

    try:
        return datetime.strptime(dt_str, "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        dt_obj = datetime.strptime(dt_str, "%Y-%m-%d")
        if is_end_date:
            return dt_obj.replace(hour=23, minute=59, second=59)
        return dt_obj


def is_within_time_range(value: datetime, start_dt: datetime | None, end_dt: datetime | None) -> bool:
    if start_dt is not None and value < start_dt:
        return False
    if end_dt is not None and value > end_dt:
        return False
    return True


def setup_plotting_style(plot_style: str) -> None:
    if plot_style == "scienceplots":
        setup_scienceplots_plotting_style()
        return

    if plot_style == "corl":
        setup_corl_plotting_style()
        return

    raise ValueError(f"Unsupported plot_style: {plot_style}")


def setup_scienceplots_plotting_style() -> None:
    # This intentionally keeps the original SciencePlots-based approach available.
    # The style stack and large font defaults mirror the earlier script, while the
    # plotting functions below keep the newer wording of titles and axis labels.
    try:
        plt.style.use(["science", "ieee"])
    except OSError:
        logging.warning("Falling back to Matplotlib default style because SciencePlots styles were unavailable.")
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.size": 28,
            "axes.labelsize": 28,
            "axes.titlesize": 28,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 28,
            "figure.titlesize": 28,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def setup_corl_plotting_style() -> None:
    # Keep SciencePlots' compact scientific defaults, but avoid the IEEE preset because CoRL/PMLR
    # papers are not IEEE two-column papers. The no-latex style keeps this script usable on machines
    # without a full LaTeX installation while retaining Computer-Modern-like mathtext.
    try:
        plt.style.use(["science", "no-latex"])
    except OSError:
        logging.warning("Falling back to Matplotlib default style because SciencePlots styles were unavailable.")
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "figure.titlesize": 8.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.2,
            "legend.columnspacing": 0.8,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.prop_cycle": cycler(color=PAPER_COLORS),
        }
    )


def get_repo_root() -> Path:
    # This script is assumed to live in $REPO_ROOT/scripts.
    return Path(__file__).resolve().parent.parent


def create_output_directory(base_name: str, labels: list[str]) -> Path:
    repo_root = get_repo_root()
    logs_root = repo_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label_part = "__".join(sanitize_filename(label) for label in labels[:6])
    if len(labels) > 6:
        label_part = f"{label_part}__and_{len(labels) - 6}_more"

    dir_name = f"{sanitize_filename(base_name)}__{label_part}__{timestamp}"
    output_dir = logs_root / dir_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def setup_logging(log_level: str, output_dir: Path) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "analysis.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("wandb").setLevel(logging.WARNING)


def log_runtime_context(args: argparse.Namespace, output_dir: Path) -> None:
    script_path = Path(__file__).resolve()
    original_cwd = Path.cwd()
    python_executable = Path(sys.executable).resolve()

    rerun_command = shlex.join([str(python_executable), str(script_path)] + sys.argv[1:])
    rerun_command_from_original_cwd = f"cd {shlex.quote(str(original_cwd))} && {rerun_command}"

    args_dict = vars(args).copy()
    args_dict["metrics_summary_root_dir"] = str(args.metrics_summary_root_dir)

    invocation = {
        "timestamp": datetime.now().isoformat(),
        "argv": sys.argv,
        "cwd": str(original_cwd),
        "script_path": str(script_path),
        "python_executable": str(python_executable),
        "python_version": sys.version,
        "platform": platform.platform(),
        "repo_root": str(get_repo_root()),
        "output_dir": str(output_dir),
        "rerun_command": rerun_command,
        "rerun_command_from_original_cwd": rerun_command_from_original_cwd,
        "args": args_dict,
    }

    with open(output_dir / "run_invocation.json", "w", encoding="utf-8") as handle:
        json.dump(invocation, handle, indent=2, sort_keys=True)

    logging.info("Output directory: %s", output_dir)
    logging.info("Current working directory: %s", original_cwd)
    logging.info("Script path: %s", script_path)
    logging.info("Python executable: %s", python_executable)
    logging.info("Python version: %s", sys.version.replace("\n", " "))
    logging.info("Platform: %s", platform.platform())
    logging.info("Repo root: %s", get_repo_root())
    logging.info("Raw argv: %s", sys.argv)
    logging.info("Rerun command: %s", rerun_command)
    logging.info("Rerun command from original cwd: %s", rerun_command_from_original_cwd)
    logging.info("Parsed arguments:")
    for key, value in sorted(args_dict.items()):
        logging.info("  %s = %s", key, value)


def make_wandb_path(env_name: str, wandb_entity: str | None) -> str:
    if wandb_entity:
        return f"{wandb_entity}/{env_name}"
    return env_name


def compute_tail_mean(series: pd.Series, tail_points: int) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return np.nan
    count = max(1, min(int(tail_points), len(cleaned)))
    return float(cleaned.iloc[-count:].mean())


def compute_mean_sem_ci95(values: pd.Series) -> tuple[float, float, float, int]:
    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(cleaned))
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(cleaned.mean())
    if n == 1:
        return mean, np.nan, np.nan, 1
    sem = float(cleaned.std(ddof=1) / np.sqrt(n))
    ci95 = 1.96 * sem
    return mean, sem, ci95, n


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def export_plot(fig: plt.Figure, output_dir: Path, stem: str, export_formats: list[str]) -> None:
    for export_format in export_formats:
        fig.savefig(
            output_dir / f"{stem}.{export_format}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
        )


def get_series_plot_style(index: int, with_marker: bool, plot_style: str) -> dict[str, Any]:
    if plot_style == "scienceplots":
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", PAPER_COLORS)
        if not colors:
            colors = PAPER_COLORS

        style: dict[str, Any] = {
            "color": colors[index % len(colors)],
            "linestyle": "-",
        }
        if with_marker:
            style["marker"] = "o"
        return style

    style = {
        "color": PAPER_COLORS[index % len(PAPER_COLORS)],
        "linestyle": PAPER_LINESTYLES[index % len(PAPER_LINESTYLES)],
    }
    if with_marker:
        style["marker"] = PAPER_MARKERS[index % len(PAPER_MARKERS)]
        style["markerfacecolor"] = "white"
        style["markeredgewidth"] = 0.8
    return style


def format_iteration_tick(value: float, _position: int) -> str:
    if abs(value) >= 1000:
        scaled = value / 1000.0
        if abs(scaled - round(scaled)) < 1e-8:
            return f"{int(round(scaled))}k"
        return f"{scaled:.1f}k"
    if abs(value - round(value)) < 1e-8:
        return f"{int(round(value))}"
    return f"{value:g}"


def as_float_array(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def format_paper_axis(
    ax: plt.Axes,
    grid_alpha: float,
    enable_minor_ticks: bool = True,
    hide_top_right_spines: bool = True,
) -> None:
    if enable_minor_ticks:
        ax.minorticks_on()

    ax.grid(True, which="major", axis="both", color="0.82", linewidth=0.45, alpha=grid_alpha)
    ax.grid(True, which="minor", axis="both", color="0.90", linewidth=0.30, alpha=0.65 * grid_alpha)
    ax.tick_params(axis="both", which="both", direction="out")

    if hide_top_right_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def format_plot_axis(ax: plt.Axes, grid_alpha: float, plot_style: str) -> None:
    if plot_style == "scienceplots":
        ax.tick_params(axis="both", which="both", direction="out")
        return

    format_paper_axis(ax, grid_alpha=grid_alpha)


def add_paper_legend(ax: plt.Axes, legend_columns: int, legend_location: str) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    ncol = min(max(1, legend_columns), len(handles))
    common_kwargs = {
        "ncol": ncol,
        "frameon": False,
        "columnspacing": 0.8,
        "handlelength": 1.9,
        "handletextpad": 0.4,
        "borderaxespad": 0.2,
    }

    if legend_location == "auto":
        if len(handles) > 4:
            ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                **common_kwargs,
            )
        else:
            ax.legend(handles, labels, loc="best", **common_kwargs)
    else:
        ax.legend(handles, labels, loc=legend_location, **common_kwargs)


def add_plot_legend(ax: plt.Axes, legend_columns: int, legend_location: str, plot_style: str) -> None:
    if plot_style == "scienceplots":
        if legend_location == "auto":
            ax.legend()
        else:
            ax.legend(loc=legend_location, ncol=max(1, legend_columns))
        return

    add_paper_legend(ax, legend_columns=legend_columns, legend_location=legend_location)


def fill_ci_band(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    ci95: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    finite_mask = np.isfinite(x) & np.isfinite(mean) & np.isfinite(ci95)
    if not finite_mask.any():
        return

    ax.fill_between(
        x[finite_mask],
        mean[finite_mask] - ci95[finite_mask],
        mean[finite_mask] + ci95[finite_mask],
        color=color,
        alpha=alpha,
        linewidth=0.0,
    )


def extract_checkpoint_from_path(path: Path) -> int | None:
    pattern = re.compile(r"^eval_checkpoint_(\d+)(?:_seed_\d+)?$")
    for part in reversed(path.parts):
        match = pattern.match(part)
        if match:
            return int(match.group(1))
    return None


def parse_cot_dataframe(metrics_summary: dict[str, Any], cot_scenario_pattern: re.Pattern[str]) -> pd.DataFrame | None:
    scenarios = metrics_summary.get("fixed_command_scenarios_metrics", {})
    if not isinstance(scenarios, dict):
        return None

    records: list[dict[str, float]] = []
    for key, metrics in scenarios.items():
        if not isinstance(metrics, dict):
            continue
        match = cot_scenario_pattern.match(key)
        if not match:
            continue

        cot_value = metrics.get("cost_of_transport")
        if cot_value is None:
            continue

        try:
            velocity = float(match.group(1))
            cot = float(cot_value)
        except (TypeError, ValueError):
            continue

        records.append(
            {
                "velocity": velocity,
                "cost_of_transport": cot,
            }
        )

    if not records:
        return None

    df = pd.DataFrame(records)
    df = df.groupby("velocity", as_index=False)["cost_of_transport"].mean()
    df.sort_values("velocity", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def parse_json_summary(
    metrics_summary: dict[str, Any],
    cot_velocity_range: tuple[float, float],
    cot_df: pd.DataFrame | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    summary["rms_error_x"] = metrics_summary.get("base_linear_velocity_x_rms_error", np.nan)
    summary["rms_error_y"] = metrics_summary.get("base_linear_velocity_y_rms_error", np.nan)

    violations = metrics_summary.get("constraint_violations_percent", {})
    torque_dict = violations.get("joint_torque") if isinstance(violations, dict) else None
    accel_dict = violations.get("joint_acceleration") if isinstance(violations, dict) else None

    if isinstance(torque_dict, dict) and torque_dict:
        summary["violation_torque"] = max(torque_dict.values())
    else:
        summary["violation_torque"] = np.nan

    if isinstance(accel_dict, dict) and accel_dict:
        summary["violation_accel"] = max(accel_dict.values())
    else:
        summary["violation_accel"] = np.nan

    min_vel, max_vel = cot_velocity_range
    if cot_df is None or cot_df.empty:
        summary["mean_cost_of_transport_range"] = np.nan
    else:
        in_range = cot_df[cot_df["velocity"].between(min_vel, max_vel)]
        if in_range.empty:
            summary["mean_cost_of_transport_range"] = np.nan
        else:
            summary["mean_cost_of_transport_range"] = float(in_range["cost_of_transport"].mean())

    return summary


def discover_metrics_summary_files(
    root_dir: Path,
    cot_scenario_pattern: str,
    cot_velocity_range: tuple[float, float],
) -> tuple[dict[tuple[str, str], JsonRunData], pd.DataFrame]:
    pattern = re.compile(cot_scenario_pattern)
    candidates: dict[tuple[str, str], list[JsonRunData]] = {}
    manifest_rows: list[dict[str, Any]] = []

    logging.info("Recursively discovering metrics_summary.json under: %s", root_dir)

    for json_path in sorted(root_dir.rglob("metrics_summary.json")):
        logging.info("  Found metrics_summary.json: %s", json_path)
        with open(json_path, "r", encoding="utf-8") as handle:
            metrics_summary = json.load(handle)

        env_name = metrics_summary.get("env_name")
        run_name = metrics_summary.get("run_name")

        if not isinstance(env_name, str) or not env_name:
            logging.warning("Skipping metrics_summary.json without valid env_name: %s", json_path)
            continue
        if not isinstance(run_name, str) or not run_name:
            logging.warning("Skipping metrics_summary.json without valid run_name: %s", json_path)
            continue

        checkpoint = extract_checkpoint_from_path(json_path)
        cot_df = parse_cot_dataframe(metrics_summary, pattern)
        summary = parse_json_summary(metrics_summary, cot_velocity_range, cot_df)
        summary["run_name"] = run_name
        summary["env_name"] = env_name
        summary["json_path"] = str(json_path)
        summary["checkpoint"] = checkpoint

        run_data = JsonRunData(
            env_name=env_name,
            run_name=run_name,
            checkpoint=checkpoint,
            json_path=json_path,
            cot_df=cot_df,
            summary=summary,
            metrics_summary=metrics_summary,
        )

        key = (env_name, run_name)
        candidates.setdefault(key, []).append(run_data)

        manifest_rows.append(
            {
                "env_name": env_name,
                "run_name": run_name,
                "checkpoint": checkpoint,
                "json_path": str(json_path),
            }
        )

    selected: dict[tuple[str, str], JsonRunData] = {}

    for key, entries in sorted(candidates.items()):
        if len(entries) == 1:
            selected[key] = entries[0]
            continue

        parseable = [entry for entry in entries if entry.checkpoint is not None]
        if not parseable:
            paths = [str(entry.json_path) for entry in entries]
            raise ValueError(
                f"Multiple metrics_summary.json files found for env_name={key[0]!r}, run_name={key[1]!r}, "
                f"but no checkpoint could be inferred: {paths}"
            )

        max_checkpoint = max(entry.checkpoint for entry in parseable if entry.checkpoint is not None)
        latest_entries = [entry for entry in parseable if entry.checkpoint == max_checkpoint]

        if len(latest_entries) != 1:
            paths = [str(entry.json_path) for entry in latest_entries]
            raise ValueError(
                f"Multiple metrics_summary.json files found for env_name={key[0]!r}, run_name={key[1]!r} "
                f"at latest checkpoint {max_checkpoint}: {paths}"
            )

        logging.warning(
            "Multiple metrics_summary.json files found for env_name=%s, run_name=%s. Using latest checkpoint %s: %s",
            key[0],
            key[1],
            max_checkpoint,
            latest_entries[0].json_path,
        )
        selected[key] = latest_entries[0]

    manifest_df = pd.DataFrame(manifest_rows)
    if not manifest_df.empty:
        manifest_df.sort_values(["env_name", "run_name", "checkpoint", "json_path"], inplace=True)

    logging.info("Discovered %d usable metrics_summary.json files.", len(manifest_rows))
    logging.info("Selected %d unique (env_name, run_name) JSON entries after checkpoint resolution.", len(selected))

    return selected, manifest_df


def fetch_wandb_runs(
    wandb_path: str,
    start_dt: datetime | None,
    end_dt: datetime | None,
    metrics: list[str],
    min_iterations: int,
    summary_tail_points: int,
) -> dict[str, dict[str, Any]]:
    api = wandb.Api()
    runs = api.runs(wandb_path)

    logging.info("Accessing WandB project: %s", wandb_path)

    result: dict[str, dict[str, Any]] = {}
    required_keys = ["_step"] + metrics

    for run in runs:
        run_dt = parse_run_name_to_datetime(run.name)
        if run_dt is None or not is_within_time_range(run_dt, start_dt, end_dt):
            continue

        logging.info("  Fetching WandB run: %s", run.name)
        history_df = pd.DataFrame(run.scan_history(keys=required_keys))

        if history_df.empty:
            logging.warning("    Skipping WandB run '%s': empty history.", run.name)
            continue

        history_df = history_df.rename(columns={"_step": "iteration"})

        if len(history_df) < min_iterations:
            logging.warning(
                "    Skipping WandB run '%s': too few history rows (%d < %d).",
                run.name,
                len(history_df),
                min_iterations,
            )
            continue

        summary: dict[str, Any] = {
            "run_name": run.name,
            "wandb_path": wandb_path,
        }

        for metric in metrics:
            if metric not in history_df.columns:
                continue
            summary[metric] = compute_tail_mean(history_df[metric], summary_tail_points)

        result[run.name] = {
            "history": history_df,
            "summary": summary,
        }

    logging.info("Found %d valid WandB runs for project '%s'.", len(result), wandb_path)
    return result


def filter_json_runs_for_series(
    json_index: dict[tuple[str, str], JsonRunData],
    spec: SeriesSpec,
) -> dict[str, JsonRunData]:
    result: dict[str, JsonRunData] = {}

    for (env_name, run_name), run_data in json_index.items():
        if env_name != spec.env_name:
            continue

        run_dt = parse_run_name_to_datetime(run_name)
        if not run_dt:
            logging.warning(
                "Skipping JSON run because run_name is not parseable as datetime: env_name=%s run_name=%s path=%s",
                env_name,
                run_name,
                run_data.json_path,
            )
            continue

        if not is_within_time_range(run_dt, spec.start_dt, spec.end_dt):
            continue

        result[run_name] = run_data

    logging.info(
        "Selected %d JSON runs for label='%s', env_name='%s', time range=%s to %s.",
        len(result),
        spec.label,
        spec.env_name,
        spec.start_dt if spec.start_dt is not None else ALL_TIME_PLACEHOLDER,
        spec.end_dt if spec.end_dt is not None else ALL_TIME_PLACEHOLDER,
    )
    return result


def resolve_selected_run_names(
    wandb_runs: dict[str, dict[str, Any]],
    json_runs: dict[str, JsonRunData],
    match_mode: str,
) -> tuple[list[str], list[str]]:
    wandb_names = sorted(wandb_runs.keys())
    json_names = sorted(json_runs.keys())

    if match_mode == "independent":
        return wandb_names, json_names

    if match_mode == "intersection":
        common = sorted(set(wandb_names).intersection(json_names))
        return common, common

    raise ValueError(f"Unsupported match_mode: {match_mode}")


def get_history_max_iteration(history: pd.DataFrame) -> int | None:
    if history.empty or "iteration" not in history.columns:
        return None

    iterations = pd.to_numeric(history["iteration"], errors="coerce").dropna()
    if iterations.empty:
        return None

    return int(iterations.max())


def compute_global_shortest_selected_wandb_iteration(series_data: dict[str, SeriesData]) -> int | None:
    max_iterations: list[int] = []

    for data in series_data.values():
        for run_name in data.selected_wandb_run_names:
            run_entry = data.wandb_runs.get(run_name)
            if run_entry is None:
                continue

            history = run_entry.get("history")
            if not isinstance(history, pd.DataFrame):
                continue

            max_iteration = get_history_max_iteration(history)
            if max_iteration is None:
                continue

            max_iterations.append(max_iteration)

    if not max_iterations:
        return None

    return min(max_iterations)


def truncate_history_to_max_iteration(history: pd.DataFrame, max_iteration: int | None) -> pd.DataFrame:
    if max_iteration is None or history.empty or "iteration" not in history.columns:
        return history.copy()

    truncated = history.copy()
    truncated["iteration"] = pd.to_numeric(truncated["iteration"], errors="coerce")
    truncated = truncated.dropna(subset=["iteration"])
    truncated = truncated[truncated["iteration"] <= max_iteration].copy()
    return truncated


def align_timeseries_data(run_histories: list[pd.DataFrame], metric_name: str) -> pd.DataFrame:
    if not run_histories:
        return pd.DataFrame(columns=["iteration", "mean", "std", "count", "sem", "ci95"])

    aligned: list[pd.DataFrame] = []

    for run_index, history in enumerate(run_histories):
        if metric_name not in history.columns:
            continue

        metric_df = history[["iteration", metric_name]].copy()
        metric_df = metric_df.dropna(subset=[metric_name])

        if metric_df.empty:
            continue

        metric_df["iteration"] = pd.to_numeric(metric_df["iteration"], errors="coerce")
        metric_df[metric_name] = pd.to_numeric(metric_df[metric_name], errors="coerce")
        metric_df = metric_df.dropna(subset=["iteration", metric_name])

        if metric_df.empty:
            continue

        metric_df = metric_df.groupby("iteration", as_index=False)[metric_name].mean()
        metric_df = metric_df.set_index("iteration")
        metric_df = metric_df.rename(columns={metric_name: f"run_{run_index}"})
        aligned.append(metric_df)

    if not aligned:
        return pd.DataFrame(columns=["iteration", "mean", "std", "count", "sem", "ci95"])

    combined = pd.concat(aligned, axis=1, join="outer").sort_index()

    stats = pd.DataFrame(index=combined.index)
    stats["mean"] = combined.mean(axis=1)
    stats["std"] = combined.std(axis=1)
    stats["count"] = combined.count(axis=1)
    stats["sem"] = stats["std"] / np.sqrt(stats["count"].where(stats["count"] > 0, np.nan))
    stats["ci95"] = 1.96 * stats["sem"]
    stats = stats.dropna(subset=["mean"])
    stats = stats.reset_index().rename(columns={"index": "iteration"})
    return stats


def filter_timeseries_for_plotting(stats: pd.DataFrame, skip_initial_iterations: int) -> pd.DataFrame:
    if stats.empty:
        return stats
    return stats[stats["iteration"] >= skip_initial_iterations].copy()


def aggregate_cot_data(run_entries: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    if not run_entries:
        return pd.DataFrame(columns=["velocity", "mean", "std", "count", "sem", "ci95"])

    rows: list[pd.DataFrame] = []
    for run_name, cot_df in run_entries:
        if cot_df is None or cot_df.empty:
            continue
        df = cot_df.copy()
        df["run_name"] = run_name
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["velocity", "mean", "std", "count", "sem", "ci95"])

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.groupby(["run_name", "velocity"], as_index=False)["cost_of_transport"].mean()

    stats = combined.groupby("velocity")["cost_of_transport"].agg(["mean", "std", "count"]).reset_index()
    stats["sem"] = stats["std"] / np.sqrt(stats["count"].where(stats["count"] > 0, np.nan))
    stats["ci95"] = 1.96 * stats["sem"]
    stats.sort_values("velocity", inplace=True)
    return stats


def filter_cot_stats_by_velocity_range(
    stats: pd.DataFrame,
    min_velocity: float | None,
    max_velocity: float | None,
) -> pd.DataFrame:
    if stats.empty:
        return stats.copy()

    filtered = stats.copy()

    if min_velocity is not None:
        filtered = filtered[filtered["velocity"] >= min_velocity]

    if max_velocity is not None:
        filtered = filtered[filtered["velocity"] <= max_velocity]

    filtered = filtered.sort_values("velocity").reset_index(drop=True)
    return filtered


def build_per_run_summary(series_data: dict[str, SeriesData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for label, data in series_data.items():
        all_run_names = sorted(set(data.wandb_runs.keys()).union(data.json_runs.keys()))
        for run_name in all_run_names:
            row: dict[str, Any] = {
                "label": label,
                "env_name": data.env_name,
                "run_name": run_name,
                "has_wandb": run_name in data.wandb_runs,
                "has_local_json": run_name in data.json_runs,
                "selected_for_wandb": run_name in data.selected_wandb_run_names,
                "selected_for_local_json": run_name in data.selected_json_run_names,
            }

            if run_name in data.wandb_runs:
                row.update(data.wandb_runs[run_name]["summary"])

            if run_name in data.json_runs:
                row.update(data.json_runs[run_name].summary)

            rms_values = [row.get("rms_error_x", np.nan), row.get("rms_error_y", np.nan)]
            rms_values = [value for value in rms_values if not pd.isna(value)]
            row["rms_error_xy_mean"] = float(np.mean(rms_values)) if rms_values else np.nan

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["label", "run_name"], inplace=True)
    return df


def build_aggregate_summary(
    per_run_df: pd.DataFrame,
    summary_wandb_metrics: list[str],
    summary_json_metrics: list[str],
) -> pd.DataFrame:
    if per_run_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for label, group in per_run_df.groupby("label"):
        for metric in summary_wandb_metrics:
            if metric not in group.columns:
                continue
            metric_group = group[group["selected_for_wandb"]]
            mean, sem, ci95, n = compute_mean_sem_ci95(metric_group[metric])
            rows.append(
                {
                    "label": label,
                    "metric": metric,
                    "display_name": DEFAULT_SUMMARY_LABEL_MAP.get(metric, metric),
                    "source": "wandb",
                    "mean": mean,
                    "sem": sem,
                    "ci95": ci95,
                    "n": n,
                }
            )

        for metric in summary_json_metrics:
            if metric not in group.columns:
                continue
            metric_group = group[group["selected_for_local_json"]]
            mean, sem, ci95, n = compute_mean_sem_ci95(metric_group[metric])
            rows.append(
                {
                    "label": label,
                    "metric": metric,
                    "display_name": DEFAULT_SUMMARY_LABEL_MAP.get(metric, metric),
                    "source": "local_json",
                    "mean": mean,
                    "sem": sem,
                    "ci95": ci95,
                    "n": n,
                }
            )

    df = pd.DataFrame(rows)
    df.sort_values(["label", "source", "metric"], inplace=True)
    return df


def save_text_summary(
    aggregate_df: pd.DataFrame,
    output_dir: Path,
    summary_tail_points: int,
    match_mode: str,
    cot_velocity_range: tuple[float, float],
    cot_filtered_plot_min_velocity: float | None,
    cot_filtered_plot_max_velocity: float | None,
    truncate_wandb_timeseries_to_shortest_run: bool,
    wandb_timeseries_truncation_iteration: int | None,
    plot_style: str,
) -> None:
    lines: list[str] = []
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"match_mode: {match_mode}")
    lines.append(f"plot_style: {plot_style}")
    lines.append(f"summary_tail_points: {summary_tail_points}")
    lines.append(f"cot_velocity_range: {cot_velocity_range[0]} to {cot_velocity_range[1]}")
    lines.append(
        "cot_filtered_plot_velocity_range: "
        f"{cot_filtered_plot_min_velocity if cot_filtered_plot_min_velocity is not None else '-inf'} "
        f"to {cot_filtered_plot_max_velocity if cot_filtered_plot_max_velocity is not None else 'inf'}"
    )
    lines.append(
        f"truncate_wandb_timeseries_to_shortest_run: {truncate_wandb_timeseries_to_shortest_run}"
    )
    lines.append(
        "wandb_timeseries_truncation_iteration: "
        f"{wandb_timeseries_truncation_iteration if wandb_timeseries_truncation_iteration is not None else 'Not applied'}"
    )
    lines.append("")

    if aggregate_df.empty:
        lines.append("No aggregate statistics available.")
    else:
        for label, group in aggregate_df.groupby("label"):
            lines.append(f"Series: {label}")
            for _, row in group.iterrows():
                display_name = row["display_name"]
                source = row["source"]
                if row["n"] == 0 or pd.isna(row["mean"]):
                    lines.append(f"  [{source}] {display_name}: Not available")
                elif row["n"] == 1 or pd.isna(row["ci95"]):
                    lines.append(f"  [{source}] {display_name}: {row['mean']:.6f} (n=1)")
                else:
                    lines.append(
                        f"  [{source}] {display_name}: {row['mean']:.6f} ± {row['ci95']:.6f} "
                        f"(95% CI, n={int(row['n'])})"
                    )
            lines.append("")

    with open(output_dir / "summary_statistics.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def plot_timeseries(
    metric_name: str,
    plot_series: list[dict[str, Any]],
    output_dir: Path,
    export_formats: list[str],
    figure_width: float,
    figure_height: float,
    show_titles: bool,
    legend_columns: int,
    legend_location: str,
    ci_alpha: float,
    grid_alpha: float,
    plot_style: str,
) -> None:
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    linewidth = 1.55 if metric_name == "Curriculum/terrain_levels" else 1.35
    ylabel = DEFAULT_LABEL_MAP.get(metric_name, metric_name.split("/")[-1].replace("_", " ").title())

    plotted_any = False

    for index, series in enumerate(plot_series):
        stats = series["stats"]
        if stats.empty:
            logging.warning("No data to plot for series '%s' and metric '%s'.", series["label"], metric_name)
            continue

        x = as_float_array(stats["iteration"])
        mean = as_float_array(stats["mean"])
        ci95 = as_float_array(stats["ci95"])
        finite_line_mask = np.isfinite(x) & np.isfinite(mean)
        if not finite_line_mask.any():
            logging.warning("No finite data to plot for series '%s' and metric '%s'.", series["label"], metric_name)
            continue

        plotted_any = True
        style = get_series_plot_style(index, with_marker=False, plot_style=plot_style)
        color = style["color"]

        ax.plot(
            x[finite_line_mask],
            mean[finite_line_mask],
            label=series["label"],
            linewidth=linewidth,
            solid_capstyle="round",
            **style,
        )
        fill_ci_band(ax, x, mean, ci95, color=color, alpha=ci_alpha)

    if not plotted_any:
        plt.close(fig)
        return

    if show_titles:
        ax.set_title(f"{ylabel} progression")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_iteration_tick))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune=None))
    ax.margins(x=0.01)

    if metric_name == "Episode/EnergyConsumed":
        ax.set_ylim(0, 2000)

    format_plot_axis(ax, grid_alpha=grid_alpha, plot_style=plot_style)
    add_plot_legend(ax, legend_columns=legend_columns, legend_location=legend_location, plot_style=plot_style)

    export_plot(fig, output_dir, f"plot_wandb_{sanitize_filename(metric_name)}", export_formats)
    plt.close(fig)


def plot_cot_comparison(
    plot_series: list[dict[str, Any]],
    output_dir: Path,
    export_formats: list[str],
    plot_baseline: bool,
    figure_width: float,
    figure_height: float,
    show_titles: bool,
    legend_columns: int,
    legend_location: str,
    ci_alpha: float,
    grid_alpha: float,
    plot_style: str,
    output_stem: str = "plot_cot_sweep",
    title: str = "Cost of transport velocity sweep",
    velocity_min: float | None = None,
    velocity_max: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    plotted_any = False

    for index, series in enumerate(plot_series):
        stats = filter_cot_stats_by_velocity_range(
            series["stats"],
            min_velocity=velocity_min,
            max_velocity=velocity_max,
        )

        if stats.empty:
            logging.warning(
                "No CoT data to plot for series '%s' after applying velocity range [%s, %s].",
                series["label"],
                velocity_min,
                velocity_max,
            )
            continue

        x = as_float_array(stats["velocity"])
        mean = as_float_array(stats["mean"])
        ci95 = as_float_array(stats["ci95"])
        finite_line_mask = np.isfinite(x) & np.isfinite(mean)
        if not finite_line_mask.any():
            logging.warning("No finite CoT data to plot for series '%s'.", series["label"])
            continue

        plotted_any = True
        style = get_series_plot_style(index, with_marker=True, plot_style=plot_style)
        color = style["color"]

        ax.plot(
            x[finite_line_mask],
            mean[finite_line_mask],
            label=series["label"],
            linewidth=1.55,
            markersize=4.4,
            solid_capstyle="round",
            **style,
        )
        fill_ci_band(ax, x, mean, ci95, color=color, alpha=ci_alpha)

    if plot_baseline:
        baseline_df = pd.DataFrame(
            {
                "velocity": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
                "cost_of_transport": [0.92, 0.75, 0.65, 0.60, 0.55, 0.50],
            }
        )

        baseline_df = filter_cot_stats_by_velocity_range(
            baseline_df,
            min_velocity=velocity_min,
            max_velocity=velocity_max,
        )

        if not baseline_df.empty:
            ax.plot(
                baseline_df["velocity"],
                baseline_df["cost_of_transport"],
                label="Baseline (ref.)",
                color=BASELINE_COLOR,
                linestyle=(0, (4, 2)),
                marker="s",
                markerfacecolor="white",
                markeredgewidth=0.8,
                linewidth=1.45,
                markersize=4.2,
            )
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    if show_titles:
        ax.set_title(title)
    ax.set_xlabel(r"Commanded velocity ($\mathrm{m\,s^{-1}}$)")
    ax.set_ylabel("Cost of transport")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune=None))
    ax.margins(x=0.02, y=0.08)

    current_left, current_right = ax.get_xlim()
    if velocity_min is not None:
        current_left = velocity_min
    if velocity_max is not None:
        current_right = velocity_max
    if current_left < current_right:
        ax.set_xlim(left=current_left, right=current_right)

    format_plot_axis(ax, grid_alpha=grid_alpha, plot_style=plot_style)
    add_plot_legend(ax, legend_columns=legend_columns, legend_location=legend_location, plot_style=plot_style)

    export_plot(fig, output_dir, output_stem, export_formats)
    plt.close(fig)


def resolve_sim_data_path(json_path: Path) -> Path:
    return json_path.parent / "plots" / "sim_data.npz"


def extract_scenario_tag(scenario_entry: Any) -> str | None:
    if isinstance(scenario_entry, (list, tuple)) and scenario_entry:
        return scenario_entry[0] if isinstance(scenario_entry[0], str) else None
    if isinstance(scenario_entry, dict):
        scenario_tag = scenario_entry.get("tag")
        return scenario_tag if isinstance(scenario_tag, str) else None
    return None


def infer_fixed_command_scenario_ranges(metrics_summary: dict[str, Any]) -> dict[str, tuple[int, int]]:
    scenarios = metrics_summary.get("fixed_command_scenarios")
    random_sim_steps = metrics_summary.get("random_sim_steps")
    total_sim_steps = metrics_summary.get("total_sim_steps")

    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("metrics_summary does not contain a valid non-empty 'fixed_command_scenarios' list.")
    if not isinstance(random_sim_steps, (int, float)):
        raise ValueError("metrics_summary does not contain a valid 'random_sim_steps' value.")
    if not isinstance(total_sim_steps, (int, float)):
        raise ValueError("metrics_summary does not contain a valid 'total_sim_steps' value.")

    random_sim_steps_int = int(random_sim_steps)
    total_sim_steps_int = int(total_sim_steps)
    if random_sim_steps_int < 0 or total_sim_steps_int <= random_sim_steps_int:
        raise ValueError(
            f"Invalid total/random simulation step counts in metrics_summary: "
            f"random_sim_steps={random_sim_steps_int}, total_sim_steps={total_sim_steps_int}"
        )

    fixed_command_total_steps = total_sim_steps_int - random_sim_steps_int
    if fixed_command_total_steps <= 0:
        raise ValueError("No fixed-command steps available in metrics_summary.")

    fixed_command_steps_per_scenario_float = fixed_command_total_steps / len(scenarios)
    fixed_command_steps_per_scenario = int(round(fixed_command_steps_per_scenario_float))
    if not np.isclose(fixed_command_steps_per_scenario_float, fixed_command_steps_per_scenario):
        raise ValueError(
            "Could not infer a consistent fixed-command window length from metrics_summary: "
            f"fixed_command_total_steps={fixed_command_total_steps}, num_scenarios={len(scenarios)}"
        )

    ranges: dict[str, tuple[int, int]] = {}
    for scenario_index, scenario_entry in enumerate(scenarios):
        scenario_tag = extract_scenario_tag(scenario_entry)
        if not scenario_tag:
            raise ValueError(f"Encountered invalid fixed-command scenario entry: {scenario_entry!r}")

        if scenario_tag in ranges:
            raise ValueError(f"Duplicate fixed-command scenario tag in metrics_summary: {scenario_tag!r}")

        start_step = random_sim_steps_int + scenario_index * fixed_command_steps_per_scenario
        end_step_exclusive = start_step + fixed_command_steps_per_scenario
        ranges[scenario_tag] = (start_step, end_step_exclusive)

    return ranges


def load_sim_data_npz(sim_data_path: Path) -> dict[str, Any]:
    with np.load(sim_data_path, allow_pickle=True) as npz_file:
        return {key: npz_file[key] for key in npz_file.files}


def compute_swing_height_records_for_range(
    sim_data: dict[str, Any],
    start_step: int,
    end_step_exclusive: int,
) -> list[dict[str, Any]]:
    sim_times = np.asarray(sim_data["sim_times"])
    contact_state_array = np.asarray(sim_data["contact_state_array"])
    foot_positions_contact_frame_array = np.asarray(sim_data["foot_positions_contact_frame_array"])
    reset_times = np.asarray(sim_data["reset_times"])
    foot_labels = [str(label) for label in np.asarray(sim_data["foot_labels"]).tolist()]

    if start_step < 0 or end_step_exclusive > len(sim_times) or start_step >= end_step_exclusive:
        raise ValueError(
            f"Invalid timestep range [{start_step}, {end_step_exclusive}) for sim_data with {len(sim_times)} steps."
        )

    time_slice = slice(start_step, end_step_exclusive)
    sim_times_sliced = sim_times[time_slice]
    if len(sim_times_sliced) < 2:
        return []

    contact_state_sliced = contact_state_array[time_slice]
    foot_heights_contact_sliced = foot_positions_contact_frame_array[time_slice, :, 2]

    t0 = float(sim_times_sliced[0])
    t1 = float(sim_times_sliced[-1])
    step_dt = float(sim_times_sliced[1] - sim_times_sliced[0])

    resets_in_window = reset_times[(reset_times >= t0) & (reset_times <= t1)]
    local_reset_steps = [int(round((reset_time - t0) / step_dt)) for reset_time in resets_in_window]

    swing_heights_dict = compute_swing_heights(
        contact_state=contact_state_sliced,
        foot_heights_contact=foot_heights_contact_sliced,
        reset_steps=local_reset_steps,
        foot_labels=foot_labels,
    )

    records: list[dict[str, Any]] = []
    filtered_negative_count = 0
    filtered_nonfinite_count = 0

    for foot_label, heights in swing_heights_dict.items():
        for height in heights:
            height_value = float(height)

            if not np.isfinite(height_value):
                filtered_nonfinite_count += 1
                continue

            if height_value < 0.0:
                filtered_negative_count += 1
                continue

            records.append(
                {
                    "foot_label": foot_label,
                    "swing_height": height_value,
                }
            )

    if filtered_negative_count > 0 or filtered_nonfinite_count > 0:
        logging.debug(
            "Filtered swing-height samples for range [%d, %d): negative=%d, nonfinite=%d",
            start_step,
            end_step_exclusive,
            filtered_negative_count,
            filtered_nonfinite_count,
        )

    return records


def collect_step_height_records(
    series_data: dict[str, SeriesData],
    flat_scenario_tag: str,
    uneven_scenario_tag: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for label, data in series_data.items():
        if not data.selected_json_run_names:
            logging.warning("Skipping step-height extraction for '%s': no selected local JSON runs.", label)
            continue

        for run_name in data.selected_json_run_names:
            json_run = data.json_runs.get(run_name)
            if json_run is None:
                continue

            sim_data_path = resolve_sim_data_path(json_run.json_path)
            if not sim_data_path.is_file():
                logging.warning(
                    "Skipping step-height extraction for label='%s', run='%s': sim_data.npz not found at %s",
                    label,
                    run_name,
                    sim_data_path,
                )
                continue

            try:
                scenario_ranges = infer_fixed_command_scenario_ranges(json_run.metrics_summary)
            except Exception as exception:
                logging.warning(
                    "Skipping step-height extraction for label='%s', run='%s': failed to infer scenario ranges from %s (%s)",
                    label,
                    run_name,
                    json_run.json_path,
                    exception,
                )
                continue

            missing_tags = [tag for tag in (flat_scenario_tag, uneven_scenario_tag) if tag not in scenario_ranges]
            if missing_tags:
                logging.warning(
                    "Skipping step-height extraction for label='%s', run='%s': missing required scenario tags %s in %s",
                    label,
                    run_name,
                    missing_tags,
                    json_run.json_path,
                )
                continue

            try:
                sim_data = load_sim_data_npz(sim_data_path)
            except Exception as exception:
                logging.warning(
                    "Skipping step-height extraction for label='%s', run='%s': failed to load %s (%s)",
                    label,
                    run_name,
                    sim_data_path,
                    exception,
                )
                continue

            for terrain_condition, scenario_tag in (("flat", flat_scenario_tag), ("uneven", uneven_scenario_tag)):
                start_step, end_step_exclusive = scenario_ranges[scenario_tag]
                precomputed_scenario_metrics = json_run.metrics_summary.get("fixed_command_scenarios_metrics", {}).get(
                    scenario_tag, {}
                )
                try:
                    scenario_records = compute_swing_height_records_for_range(
                        sim_data=sim_data,
                        start_step=start_step,
                        end_step_exclusive=end_step_exclusive,
                    )
                except Exception as exception:
                    logging.warning(
                        "Skipping step-height extraction for label='%s', run='%s', scenario='%s': %s",
                        label,
                        run_name,
                        scenario_tag,
                        exception,
                    )
                    continue

                for record in scenario_records:
                    rows.append(
                        {
                            "label": label,
                            "env_name": data.env_name,
                            "run_name": run_name,
                            "terrain_condition": terrain_condition,
                            "scenario_tag": scenario_tag,
                            "json_path": str(json_run.json_path),
                            "sim_data_path": str(sim_data_path),
                            "checkpoint": json_run.checkpoint,
                            "start_step": start_step,
                            "end_step_exclusive": end_step_exclusive,
                            "foot_label": record["foot_label"],
                            "swing_height": record["swing_height"],
                            "precomputed_step_height_summary_available": isinstance(
                                precomputed_scenario_metrics.get("step_height_summary"), dict
                            ),
                        }
                    )

            logging.info(
                "Extracted step-height data for label='%s', run='%s' using flat='%s' and uneven='%s'.",
                label,
                run_name,
                flat_scenario_tag,
                uneven_scenario_tag,
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "label",
                "env_name",
                "run_name",
                "terrain_condition",
                "scenario_tag",
                "json_path",
                "sim_data_path",
                "checkpoint",
                "start_step",
                "end_step_exclusive",
                "foot_label",
                "swing_height",
                "precomputed_step_height_summary_available",
            ]
        )

    df = pd.DataFrame(rows)
    df.sort_values(["label", "run_name", "terrain_condition", "foot_label", "swing_height"], inplace=True)
    return df


def build_step_height_summary(step_height_df: pd.DataFrame) -> pd.DataFrame:
    if step_height_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    grouping_columns = ["label", "env_name", "terrain_condition", "scenario_tag"]
    for group_keys, group in step_height_df.groupby(grouping_columns):
        label, env_name, terrain_condition, scenario_tag = group_keys
        heights = pd.to_numeric(group["swing_height"], errors="coerce").dropna().to_numpy(dtype=float)
        summary = summarize_metric(heights.tolist())

        row: dict[str, Any] = {
            "label": label,
            "env_name": env_name,
            "terrain_condition": terrain_condition,
            "scenario_tag": scenario_tag,
            "num_swing_events": int(len(heights)),
            "num_runs": int(group["run_name"].nunique()),
        }
        row.update(summary)
        rows.append(row)

    paired_rows: list[dict[str, Any]] = []
    for (label, env_name), group in df_groupby_label_env(rows):
        flat_row = next((row for row in group if row["terrain_condition"] == "flat"), None)
        uneven_row = next((row for row in group if row["terrain_condition"] == "uneven"), None)
        if flat_row is not None and uneven_row is not None:
            paired_rows.append(
                {
                    "label": label,
                    "env_name": env_name,
                    "flat_mean": flat_row.get("mean", np.nan),
                    "uneven_mean": uneven_row.get("mean", np.nan),
                    "mean_difference_uneven_minus_flat": float(
                        uneven_row.get("mean", np.nan) - flat_row.get("mean", np.nan)
                    )
                    if not pd.isna(flat_row.get("mean", np.nan)) and not pd.isna(uneven_row.get("mean", np.nan))
                    else np.nan,
                    "flat_stddev": flat_row.get("stddev", np.nan),
                    "uneven_stddev": uneven_row.get("stddev", np.nan),
                    "stddev_difference_uneven_minus_flat": float(
                        uneven_row.get("stddev", np.nan) - flat_row.get("stddev", np.nan)
                    )
                    if not pd.isna(flat_row.get("stddev", np.nan)) and not pd.isna(uneven_row.get("stddev", np.nan))
                    else np.nan,
                }
            )

    df = pd.DataFrame(rows)
    df.sort_values(["label", "terrain_condition"], inplace=True)

    if paired_rows:
        paired_df = pd.DataFrame(paired_rows)
        df = df.merge(paired_df, on=["label", "env_name"], how="left")

    return df


def df_groupby_label_env(rows: list[dict[str, Any]]) -> list[tuple[tuple[str, str], list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["label"], row["env_name"])
        grouped.setdefault(key, []).append(row)
    return list(grouped.items())


def plot_step_height_box_comparison(
    step_height_df: pd.DataFrame,
    output_dir: Path,
    export_formats: list[str],
    showfliers: bool,
    figure_width: float,
    figure_height: float,
    show_titles: bool,
    legend_columns: int,
    legend_location: str,
    grid_alpha: float,
    plot_style: str,
) -> None:
    if step_height_df.empty:
        logging.warning("Skipping step-height box plot: no step-height samples were collected.")
        return

    ordered_labels = list(dict.fromkeys(step_height_df["label"].tolist()))

    datasets: list[np.ndarray] = []
    positions: list[float] = []
    box_kinds: list[str] = []
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    flat_offset = -0.18
    uneven_offset = 0.18
    box_width = 0.30

    for label_index, label in enumerate(ordered_labels):
        label_group = step_height_df[step_height_df["label"] == label]
        flat_values = pd.to_numeric(
            label_group[label_group["terrain_condition"] == "flat"]["swing_height"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        uneven_values = pd.to_numeric(
            label_group[label_group["terrain_condition"] == "uneven"]["swing_height"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        if flat_values.size == 0 and uneven_values.size == 0:
            logging.warning("Skipping step-height box entries for '%s': both terrain conditions are empty.", label)
            continue

        center = float(label_index + 1)
        tick_positions.append(center)
        tick_labels.append(label)

        if flat_values.size > 0:
            datasets.append(flat_values)
            positions.append(center + flat_offset)
            box_kinds.append("flat")
        else:
            logging.warning("No step-height samples for label='%s', condition='flat'.", label)

        if uneven_values.size > 0:
            datasets.append(uneven_values)
            positions.append(center + uneven_offset)
            box_kinds.append("uneven")
        else:
            logging.warning("No step-height samples for label='%s', condition='uneven'.", label)

    if not datasets:
        logging.warning("Skipping step-height box plot: all step-height groups were empty.")
        return

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    boxplot = ax.boxplot(
        datasets,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showmeans=True,
        showfliers=showfliers,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 3.2,
        },
        medianprops={
            "color": "black",
            "linewidth": 0.9,
        },
        whiskerprops={
            "linewidth": 0.8,
        },
        capprops={
            "linewidth": 0.8,
        },
        flierprops={
            "marker": ".",
            "markersize": 2.2,
            "markeredgecolor": "0.25",
            "alpha": 0.45,
        },
    )

    for patch, kind in zip(boxplot["boxes"], box_kinds):
        if kind == "flat":
            patch.set_facecolor("white")
            patch.set_edgecolor("0.20")
            patch.set_hatch("//")
            patch.set_alpha(1.0)
        else:
            patch.set_facecolor("0.75")
            patch.set_edgecolor("0.20")
            patch.set_hatch("")
            patch.set_alpha(1.0)
        patch.set_linewidth(0.8)

    if show_titles:
        ax.set_title("Swing-height distribution: flat vs. uneven terrain")
    ax.set_ylabel(r"Max swing height ($\mathrm{m}$)")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    if len(tick_labels) > 4:
        for tick_label in ax.get_xticklabels():
            tick_label.set_rotation(25)
            tick_label.set_ha("right")

    legend_handles = [
        Patch(facecolor="white", edgecolor="0.20", hatch="//", label="Flat terrain"),
        Patch(facecolor="0.75", edgecolor="0.20", label="Uneven terrain"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="best" if legend_location == "auto" else legend_location,
        ncol=min(legend_columns, 2),
        frameon=False if plot_style == "corl" else plt.rcParams.get("legend.frameon", True),
    )

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune=None))
    ax.margins(x=0.03, y=0.08)
    format_plot_axis(ax, grid_alpha=grid_alpha, plot_style=plot_style)

    export_plot(fig, output_dir, "plot_step_height_box_comparison", export_formats)
    plt.close(fig)


def build_series_specs(args: argparse.Namespace) -> list[SeriesSpec]:
    num_series = len(args.labels)
    if not (
        num_series
        == len(args.env_names)
        == len(args.start_times)
        == len(args.end_times)
    ):
        raise ValueError(
            "The number of --labels, --env_names, --start_times, and --end_times must be the same."
        )

    specs: list[SeriesSpec] = []
    for index in range(num_series):
        start_dt = parse_flexible_datetime_or_all(args.start_times[index])
        end_dt = parse_flexible_datetime_or_all(args.end_times[index], is_end_date=True)

        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ValueError(f"Start time is after end time for label '{args.labels[index]}'.")

        env_name = args.env_names[index]
        specs.append(
            SeriesSpec(
                label=args.labels[index],
                env_name=env_name,
                wandb_path=make_wandb_path(env_name, args.wandb_entity),
                start_dt=start_dt,
                end_dt=end_dt,
            )
        )

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified analysis script for WandB histories and recursively discovered metrics_summary.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--labels", nargs="+", required=True, help="Series labels.")
    parser.add_argument(
        "--env_names",
        nargs="+",
        required=True,
        help="Environment names. These are used both as metrics_summary.json env_name and as WandB project names.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional WandB entity. If set, WandB project paths become <entity>/<env_name>.",
    )
    parser.add_argument(
        "--start_times",
        nargs="+",
        required=True,
        help=f"Start date or datetime per series: YYYY-MM-DD or YYYY-MM-DD-HH-MM-SS, or {ALL_TIME_PLACEHOLDER} for no lower bound.",
    )
    parser.add_argument(
        "--end_times",
        nargs="+",
        required=True,
        help=f"End date or datetime per series: YYYY-MM-DD or YYYY-MM-DD-HH-MM-SS, or {ALL_TIME_PLACEHOLDER} for no upper bound.",
    )

    parser.add_argument(
        "--metrics_summary_root_dir",
        type=Path,
        required=True,
        help="Root directory that will be searched recursively for metrics_summary.json files.",
    )

    parser.add_argument(
        "--wandb_metrics",
        nargs="+",
        default=DEFAULT_WANDB_METRICS,
        help="WandB metrics to fetch and plot as time series.",
    )
    parser.add_argument(
        "--summary_wandb_metrics",
        nargs="+",
        default=DEFAULT_SUMMARY_WANDB_METRICS,
        help="WandB metrics to aggregate into final per-run and per-series summary statistics.",
    )

    parser.add_argument("--min_iterations", type=int, default=500, help="Minimum WandB history length to include a run.")
    parser.add_argument(
        "--summary_tail_points",
        type=int,
        default=500,
        help="Number of final training iterations used to compute final WandB summary metrics.",
    )
    parser.add_argument(
        "--plot_skip_initial_iterations",
        type=int,
        default=150,
        help="Skip iterations below this threshold for WandB progression plots and exported time-series CSVs.",
    )
    parser.add_argument(
        "--truncate_wandb_timeseries_to_shortest_run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If set, truncate WandB time-series histories used for progression plots and exported "
            "time-series CSVs to the maximum iteration of the globally shortest selected WandB run "
            "across all specified series. This does not affect final tail-based summary metrics. "
            "Use --no-truncate_wandb_timeseries_to_shortest_run to disable."
        ),
    )
    parser.add_argument(
        "--cot_velocity_range",
        type=float,
        nargs=2,
        default=(0.6, 1.6),
        help=(
            "Velocity range used only for the per-run summary metric "
            "'mean_cost_of_transport_range'. This does not affect which CoT sweep points are plotted."
        ),
    )
    parser.add_argument(
        "--cot_filtered_plot_min_velocity",
        type=float,
        default=0.4,
        help="Lower commanded-velocity bound for the additional filtered CoT plot.",
    )
    parser.add_argument(
        "--cot_filtered_plot_max_velocity",
        type=float,
        default=None,
        help="Optional upper commanded-velocity bound for the additional filtered CoT plot.",
    )
    parser.add_argument(
        "--cot_scenario_pattern",
        type=str,
        default=r"^cot_sweep_walk_x_flat_terrain_(\d+\.?\d*)$",
        help="Regex used to parse CoT sweep scenario keys. Group 1 must be the commanded velocity.",
    )
    parser.add_argument(
        "--match_mode",
        choices=["intersection", "independent"],
        default="intersection",
        help="How WandB and local JSON runs are matched before aggregation.",
    )
    parser.add_argument(
        "--step_height_flat_scenario_tag",
        type=str,
        default=DEFAULT_STEP_HEIGHT_FLAT_SCENARIO_TAG,
        help="Fixed-command scenario tag used for the flat-terrain step-height box plot.",
    )
    parser.add_argument(
        "--step_height_uneven_scenario_tag",
        type=str,
        default=DEFAULT_STEP_HEIGHT_UNEVEN_SCENARIO_TAG,
        help="Fixed-command scenario tag used for the uneven-terrain step-height box plot.",
    )
    parser.add_argument(
        "--step_height_showfliers",
        action="store_true",
        help="Show outliers in the step-height box plot.",
    )
    parser.add_argument(
        "--plot_style",
        choices=["corl", "scienceplots"],
        default="corl",
        help=(
            "Plot styling backend. 'corl' uses the publication-oriented style with a long "
            "color/marker/line-style cycle. 'scienceplots' uses the previous SciencePlots "
            "+ IEEE style stack while keeping the updated axis labels and titles."
        ),
    )

    parser.add_argument(
        "--figure_width",
        type=float,
        default=6.75,
        help="Figure width in inches. The default is intended for full-width single-column CoRL/PMLR-style figures.",
    )
    parser.add_argument(
        "--timeseries_figure_height",
        type=float,
        default=3.35,
        help="Figure height in inches for WandB time-series plots.",
    )
    parser.add_argument(
        "--cot_figure_height",
        type=float,
        default=3.35,
        help="Figure height in inches for CoT sweep plots.",
    )
    parser.add_argument(
        "--step_height_figure_height",
        type=float,
        default=3.55,
        help="Figure height in inches for the step-height box plot.",
    )
    parser.add_argument(
        "--plot_titles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw titles inside figure files. For papers, captions usually make internal titles redundant.",
    )
    parser.add_argument(
        "--legend_columns",
        type=int,
        default=2,
        help="Maximum number of legend columns. For more than four entries, auto legends are placed above the axes.",
    )
    parser.add_argument(
        "--legend_location",
        type=str,
        default="auto",
        help="Matplotlib legend location, or 'auto' for inside legends up to four entries and top legends above that.",
    )
    parser.add_argument(
        "--ci_alpha",
        type=float,
        default=0.14,
        help="Alpha value for 95%% CI bands.",
    )
    parser.add_argument(
        "--grid_alpha",
        type=float,
        default=0.55,
        help="Alpha value for the light plot grid.",
    )

    parser.add_argument("--output_name", type=str, default="analysis", help="Base name for the output directory.")
    parser.add_argument("--export_formats", nargs="+", default=["pdf"], help="Plot export formats such as pdf png svg.")
    parser.add_argument("--plot_baseline", action="store_true", help="Overlay the hardcoded CoT reference baseline.")
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.metrics_summary_root_dir.is_dir():
        raise FileNotFoundError(f"metrics_summary_root_dir does not exist: {args.metrics_summary_root_dir}")
    if args.summary_tail_points <= 0:
        raise ValueError("--summary_tail_points must be positive")
    if args.plot_skip_initial_iterations < 0:
        raise ValueError("--plot_skip_initial_iterations must be non-negative")
    if not args.step_height_flat_scenario_tag:
        raise ValueError("--step_height_flat_scenario_tag must be non-empty")
    if not args.step_height_uneven_scenario_tag:
        raise ValueError("--step_height_uneven_scenario_tag must be non-empty")
    if args.figure_width <= 0.0:
        raise ValueError("--figure_width must be positive")
    if args.timeseries_figure_height <= 0.0:
        raise ValueError("--timeseries_figure_height must be positive")
    if args.cot_figure_height <= 0.0:
        raise ValueError("--cot_figure_height must be positive")
    if args.step_height_figure_height <= 0.0:
        raise ValueError("--step_height_figure_height must be positive")
    if args.legend_columns <= 0:
        raise ValueError("--legend_columns must be positive")
    if not (0.0 <= args.ci_alpha <= 1.0):
        raise ValueError("--ci_alpha must be between 0 and 1")
    if not (0.0 <= args.grid_alpha <= 1.0):
        raise ValueError("--grid_alpha must be between 0 and 1")

    cot_velocity_range = (float(args.cot_velocity_range[0]), float(args.cot_velocity_range[1]))
    if not np.isfinite(cot_velocity_range[0]) or not np.isfinite(cot_velocity_range[1]):
        raise ValueError("--cot_velocity_range values must be finite")
    if cot_velocity_range[0] > cot_velocity_range[1]:
        raise ValueError("--cot_velocity_range lower bound must be <= upper bound")

    if not np.isfinite(args.cot_filtered_plot_min_velocity):
        raise ValueError("--cot_filtered_plot_min_velocity must be finite")
    if (
        args.cot_filtered_plot_max_velocity is not None
        and not np.isfinite(args.cot_filtered_plot_max_velocity)
    ):
        raise ValueError("--cot_filtered_plot_max_velocity must be finite when provided")
    if (
        args.cot_filtered_plot_max_velocity is not None
        and args.cot_filtered_plot_min_velocity > args.cot_filtered_plot_max_velocity
    ):
        raise ValueError(
            "--cot_filtered_plot_min_velocity must be <= --cot_filtered_plot_max_velocity"
        )

    output_dir = create_output_directory(args.output_name, args.labels)
    setup_logging(args.log_level, output_dir)
    setup_plotting_style(args.plot_style)
    log_runtime_context(args, output_dir)

    series_specs = build_series_specs(args)

    logging.info("Series configuration:")
    for index, spec in enumerate(series_specs, start=1):
        logging.info(
            "  %d: label=%s env_name=%s wandb_path=%s start=%s end=%s",
            index,
            spec.label,
            spec.env_name,
            spec.wandb_path,
            spec.start_dt if spec.start_dt is not None else ALL_TIME_PLACEHOLDER,
            spec.end_dt if spec.end_dt is not None else ALL_TIME_PLACEHOLDER,
        )

    json_index, discovered_json_manifest = discover_metrics_summary_files(
        root_dir=args.metrics_summary_root_dir,
        cot_scenario_pattern=args.cot_scenario_pattern,
        cot_velocity_range=cot_velocity_range,
    )
    save_dataframe(discovered_json_manifest, output_dir / "discovered_metrics_summary_files.csv")

    all_wandb_metrics = sorted(set(args.wandb_metrics).union(args.summary_wandb_metrics))
    series_data: dict[str, SeriesData] = {}
    selection_rows: list[dict[str, Any]] = []

    for index, spec in enumerate(series_specs, start=1):
        logging.info("%s", "=" * 100)
        logging.info("Processing series %d/%d: %s", index, len(series_specs), spec.label)

        wandb_runs = fetch_wandb_runs(
            wandb_path=spec.wandb_path,
            start_dt=spec.start_dt,
            end_dt=spec.end_dt,
            metrics=all_wandb_metrics,
            min_iterations=args.min_iterations,
            summary_tail_points=args.summary_tail_points,
        )

        json_runs = filter_json_runs_for_series(
            json_index=json_index,
            spec=spec,
        )

        selected_wandb_run_names, selected_json_run_names = resolve_selected_run_names(
            wandb_runs=wandb_runs,
            json_runs=json_runs,
            match_mode=args.match_mode,
        )

        missing_local = sorted(set(wandb_runs.keys()) - set(json_runs.keys()))
        missing_wandb = sorted(set(json_runs.keys()) - set(wandb_runs.keys()))

        logging.info(
            "Series '%s' selection result: %d WandB runs, %d JSON runs, %d selected WandB runs, %d selected JSON runs.",
            spec.label,
            len(wandb_runs),
            len(json_runs),
            len(selected_wandb_run_names),
            len(selected_json_run_names),
        )

        if missing_local:
            logging.warning("  Runs present in WandB but missing local JSON for '%s': %s", spec.label, missing_local)
        if missing_wandb:
            logging.warning("  Runs present in local JSON but missing WandB for '%s': %s", spec.label, missing_wandb)

        for run_name in sorted(set(wandb_runs.keys()).union(json_runs.keys())):
            json_run = json_runs.get(run_name)
            wandb_run = wandb_runs.get(run_name)
            wandb_max_iteration = None
            if wandb_run is not None:
                history = wandb_run.get("history")
                if isinstance(history, pd.DataFrame):
                    wandb_max_iteration = get_history_max_iteration(history)

            selection_rows.append(
                {
                    "label": spec.label,
                    "env_name": spec.env_name,
                    "wandb_path": spec.wandb_path,
                    "run_name": run_name,
                    "has_wandb": run_name in wandb_runs,
                    "has_local_json": run_name in json_runs,
                    "selected_for_wandb": run_name in selected_wandb_run_names,
                    "selected_for_local_json": run_name in selected_json_run_names,
                    "json_path": str(json_run.json_path) if json_run else None,
                    "checkpoint": json_run.checkpoint if json_run else None,
                    "wandb_max_iteration": wandb_max_iteration,
                }
            )

        series_data[spec.label] = SeriesData(
            label=spec.label,
            env_name=spec.env_name,
            wandb_path=spec.wandb_path,
            wandb_runs=wandb_runs,
            json_runs=json_runs,
            selected_wandb_run_names=selected_wandb_run_names,
            selected_json_run_names=selected_json_run_names,
        )

    selection_df = pd.DataFrame(selection_rows)
    if not selection_df.empty:
        selection_df.sort_values(["label", "run_name"], inplace=True)
    save_dataframe(selection_df, output_dir / "selected_runs.csv")

    wandb_timeseries_truncation_iteration: int | None = None
    if args.truncate_wandb_timeseries_to_shortest_run:
        wandb_timeseries_truncation_iteration = compute_global_shortest_selected_wandb_iteration(series_data)
        if wandb_timeseries_truncation_iteration is None:
            logging.warning(
                "WandB time-series truncation was requested, but no valid selected WandB histories were available."
            )
        else:
            logging.info(
                "Truncating WandB time-series data to the shortest selected run: max iteration = %d",
                wandb_timeseries_truncation_iteration,
            )

    per_run_df = build_per_run_summary(series_data)
    save_dataframe(per_run_df, output_dir / "per_run_summary.csv")

    summary_json_metrics = [
        "mean_cost_of_transport_range",
        "violation_torque",
        "violation_accel",
        "rms_error_x",
        "rms_error_y",
        "rms_error_xy_mean",
    ]
    aggregate_df = build_aggregate_summary(
        per_run_df=per_run_df,
        summary_wandb_metrics=args.summary_wandb_metrics,
        summary_json_metrics=summary_json_metrics,
    )
    save_dataframe(aggregate_df, output_dir / "aggregate_summary.csv")
    save_text_summary(
        aggregate_df=aggregate_df,
        output_dir=output_dir,
        summary_tail_points=args.summary_tail_points,
        match_mode=args.match_mode,
        cot_velocity_range=cot_velocity_range,
        cot_filtered_plot_min_velocity=args.cot_filtered_plot_min_velocity,
        cot_filtered_plot_max_velocity=args.cot_filtered_plot_max_velocity,
        truncate_wandb_timeseries_to_shortest_run=args.truncate_wandb_timeseries_to_shortest_run,
        wandb_timeseries_truncation_iteration=wandb_timeseries_truncation_iteration,
        plot_style=args.plot_style,
    )

    for metric_name in args.wandb_metrics:
        plot_series: list[dict[str, Any]] = []
        metric_stats_tables: list[pd.DataFrame] = []

        for label, data in series_data.items():
            histories: list[pd.DataFrame] = []
            for run_name in data.selected_wandb_run_names:
                if run_name not in data.wandb_runs:
                    continue

                history = data.wandb_runs[run_name]["history"]
                if args.truncate_wandb_timeseries_to_shortest_run:
                    history = truncate_history_to_max_iteration(history, wandb_timeseries_truncation_iteration)

                histories.append(history)

            stats = align_timeseries_data(histories, metric_name)
            stats = filter_timeseries_for_plotting(stats, args.plot_skip_initial_iterations)

            plot_series.append(
                {
                    "label": label,
                    "stats": stats,
                }
            )

            if not stats.empty:
                stats_to_save = stats.copy()
                stats_to_save.insert(0, "label", label)
                stats_to_save.insert(1, "env_name", data.env_name)
                stats_to_save.insert(2, "metric", metric_name)
                if args.truncate_wandb_timeseries_to_shortest_run:
                    stats_to_save.insert(3, "truncated_to_iteration", wandb_timeseries_truncation_iteration)
                metric_stats_tables.append(stats_to_save)

        if metric_stats_tables:
            metric_df = pd.concat(metric_stats_tables, ignore_index=True)
            save_dataframe(metric_df, output_dir / f"timeseries_{sanitize_filename(metric_name)}.csv")

        plot_timeseries(
            metric_name=metric_name,
            plot_series=plot_series,
            output_dir=output_dir,
            export_formats=args.export_formats,
            figure_width=args.figure_width,
            figure_height=args.timeseries_figure_height,
            show_titles=args.plot_titles,
            legend_columns=args.legend_columns,
            legend_location=args.legend_location,
            ci_alpha=args.ci_alpha,
            grid_alpha=args.grid_alpha,
            plot_style=args.plot_style,
        )

    cot_plot_series: list[dict[str, Any]] = []
    cot_stats_tables: list[pd.DataFrame] = []

    cot_filtered_plot_series: list[dict[str, Any]] = []
    cot_filtered_stats_tables: list[pd.DataFrame] = []

    for label, data in series_data.items():
        run_entries: list[tuple[str, pd.DataFrame]] = []

        for run_name in data.selected_json_run_names:
            json_run = data.json_runs.get(run_name)
            if json_run is None or json_run.cot_df is None or json_run.cot_df.empty:
                continue
            run_entries.append((run_name, json_run.cot_df))

        stats = aggregate_cot_data(run_entries)
        filtered_stats = filter_cot_stats_by_velocity_range(
            stats,
            min_velocity=args.cot_filtered_plot_min_velocity,
            max_velocity=args.cot_filtered_plot_max_velocity,
        )

        cot_plot_series.append(
            {
                "label": label,
                "stats": stats,
            }
        )

        cot_filtered_plot_series.append(
            {
                "label": label,
                "stats": filtered_stats,
            }
        )

        if not stats.empty:
            stats_to_save = stats.copy()
            stats_to_save.insert(0, "label", label)
            stats_to_save.insert(1, "env_name", data.env_name)
            cot_stats_tables.append(stats_to_save)

        if not filtered_stats.empty:
            filtered_stats_to_save = filtered_stats.copy()
            filtered_stats_to_save.insert(0, "label", label)
            filtered_stats_to_save.insert(1, "env_name", data.env_name)
            filtered_stats_to_save.insert(2, "plot_velocity_min", args.cot_filtered_plot_min_velocity)
            filtered_stats_to_save.insert(3, "plot_velocity_max", args.cot_filtered_plot_max_velocity)
            cot_filtered_stats_tables.append(filtered_stats_to_save)

    if cot_stats_tables:
        cot_stats_df = pd.concat(cot_stats_tables, ignore_index=True)
        save_dataframe(cot_stats_df, output_dir / "cot_sweep_stats.csv")

    if cot_filtered_stats_tables:
        cot_filtered_stats_df = pd.concat(cot_filtered_stats_tables, ignore_index=True)
        save_dataframe(cot_filtered_stats_df, output_dir / "cot_sweep_stats_filtered.csv")

    plot_cot_comparison(
        plot_series=cot_plot_series,
        output_dir=output_dir,
        export_formats=args.export_formats,
        plot_baseline=args.plot_baseline,
        figure_width=args.figure_width,
        figure_height=args.cot_figure_height,
        show_titles=args.plot_titles,
        legend_columns=args.legend_columns,
        legend_location=args.legend_location,
        ci_alpha=args.ci_alpha,
        grid_alpha=args.grid_alpha,
        plot_style=args.plot_style,
        output_stem="plot_cot_sweep",
        title="Cost of transport velocity sweep",
    )

    plot_cot_comparison(
        plot_series=cot_filtered_plot_series,
        output_dir=output_dir,
        export_formats=args.export_formats,
        plot_baseline=args.plot_baseline,
        figure_width=args.figure_width,
        figure_height=args.cot_figure_height,
        show_titles=args.plot_titles,
        legend_columns=args.legend_columns,
        legend_location=args.legend_location,
        ci_alpha=args.ci_alpha,
        grid_alpha=args.grid_alpha,
        plot_style=args.plot_style,
        output_stem="plot_cot_sweep_filtered",
        title="Cost of transport velocity sweep",
        velocity_min=args.cot_filtered_plot_min_velocity,
        velocity_max=args.cot_filtered_plot_max_velocity,
    )

    step_height_df = collect_step_height_records(
        series_data=series_data,
        flat_scenario_tag=args.step_height_flat_scenario_tag,
        uneven_scenario_tag=args.step_height_uneven_scenario_tag,
    )
    save_dataframe(step_height_df, output_dir / "step_height_boxplot_samples.csv")

    step_height_summary_df = build_step_height_summary(step_height_df)
    save_dataframe(step_height_summary_df, output_dir / "step_height_boxplot_summary.csv")

    plot_step_height_box_comparison(
        step_height_df=step_height_df,
        output_dir=output_dir,
        export_formats=args.export_formats,
        showfliers=args.step_height_showfliers,
        figure_width=args.figure_width,
        figure_height=args.step_height_figure_height,
        show_titles=args.plot_titles,
        legend_columns=args.legend_columns,
        legend_location=args.legend_location,
        grid_alpha=args.grid_alpha,
        plot_style=args.plot_style,
    )

    logging.info("Finished. Outputs written to: %s", output_dir)


if __name__ == "__main__":
    main()