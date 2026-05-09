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

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import wandb


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
    "Curriculum/terrain_levels": "Terrain Level",
    "Episode_Reward/track_lin_vel_xy_exp": "Velocity Tracking Reward",
    "Episode/MaxJointPos": "Max. Joint Position (rad)",
    "Episode/MaxAirTime": "Max. Air Time (s)",
    "Episode/EnergyConsumed": "Energy Consumed (J)",
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

ALL_TIME_PLACEHOLDER = "ALL"


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


def setup_plotting_style() -> None:
    plt.style.use(["science", "ieee"])
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
        fig.savefig(output_dir / f"{stem}.{export_format}", dpi=600, bbox_inches="tight")


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


def parse_json_summary(metrics_summary: dict[str, Any], cot_velocity_range: tuple[float, float], cot_df: pd.DataFrame | None) -> dict[str, Any]:
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

        metric_df[metric_name] = pd.to_numeric(metric_df[metric_name], errors="coerce")
        metric_df = metric_df.dropna(subset=[metric_name])

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
) -> None:
    lines: list[str] = []
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"match_mode: {match_mode}")
    lines.append(f"summary_tail_points: {summary_tail_points}")
    lines.append(f"cot_velocity_range: {cot_velocity_range[0]} to {cot_velocity_range[1]}")
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
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linewidth = 2.0 if metric_name == "Curriculum/terrain_levels" else 1.5
    ylabel = DEFAULT_LABEL_MAP.get(metric_name, metric_name.split("/")[-1].replace("_", " ").title())

    plotted_any = False

    for index, series in enumerate(plot_series):
        stats = series["stats"]
        if stats.empty:
            logging.warning("No data to plot for series '%s' and metric '%s'.", series["label"], metric_name)
            continue

        plotted_any = True
        color = colors[index % len(colors)]

        ax.plot(
            stats["iteration"],
            stats["mean"],
            label=series["label"],
            color=color,
            linewidth=linewidth,
        )

        if not np.all(np.isnan(stats["ci95"])):
            ax.fill_between(
                stats["iteration"],
                stats["mean"] - stats["ci95"],
                stats["mean"] + stats["ci95"],
                color=color,
                alpha=0.2,
            )

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_title(f"{ylabel} Progression", fontsize=34)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend()

    if metric_name == "Episode/EnergyConsumed":
        ax.set_ylim(0, 2000)

    export_plot(fig, output_dir, f"plot_wandb_{sanitize_filename(metric_name)}", export_formats)
    plt.close(fig)


def plot_cot_comparison(
    plot_series: list[dict[str, Any]],
    output_dir: Path,
    export_formats: list[str],
    plot_baseline: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plotted_any = False

    for index, series in enumerate(plot_series):
        stats = series["stats"]
        if stats.empty:
            logging.warning("No CoT data to plot for series '%s'.", series["label"])
            continue

        plotted_any = True
        color = colors[index % len(colors)]

        ax.plot(
            stats["velocity"],
            stats["mean"],
            label=series["label"],
            color=color,
            linestyle="-",
            marker="o",
            linewidth=2.5,
            markersize=8,
        )

        if not np.all(np.isnan(stats["ci95"])):
            ax.fill_between(
                stats["velocity"],
                stats["mean"] - stats["ci95"],
                stats["mean"] + stats["ci95"],
                color=color,
                alpha=0.2,
            )

    if plot_baseline:
        baseline_velocities = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        baseline_cot = [0.92, 0.75, 0.65, 0.60, 0.55, 0.50]
        ax.plot(
            baseline_velocities,
            baseline_cot,
            label="Baseline (Ref. Paper)",
            color="dimgray",
            linestyle="--",
            marker="s",
            linewidth=2.5,
            markersize=8,
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_title("Cost of Transport Velocity Sweep", fontsize=34)
    ax.set_xlabel("Commanded Velocity (m/s)")
    ax.set_ylabel("Cost of Transport")
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10, prune="both"))

    export_plot(fig, output_dir, "plot_cot_sweep", export_formats)
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
        "--cot_velocity_range",
        type=float,
        nargs=2,
        default=(0.6, 1.6),
        help="Velocity range used to compute per-run mean cost of transport summaries.",
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

    output_dir = create_output_directory(args.output_name, args.labels)
    setup_logging(args.log_level, output_dir)
    setup_plotting_style()
    log_runtime_context(args, output_dir)

    series_specs = build_series_specs(args)
    cot_velocity_range = (float(args.cot_velocity_range[0]), float(args.cot_velocity_range[1]))

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
    )

    for metric_name in args.wandb_metrics:
        plot_series: list[dict[str, Any]] = []
        metric_stats_tables: list[pd.DataFrame] = []

        for label, data in series_data.items():
            histories = [
                data.wandb_runs[run_name]["history"]
                for run_name in data.selected_wandb_run_names
                if run_name in data.wandb_runs
            ]
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
                metric_stats_tables.append(stats_to_save)

        if metric_stats_tables:
            metric_df = pd.concat(metric_stats_tables, ignore_index=True)
            save_dataframe(metric_df, output_dir / f"timeseries_{sanitize_filename(metric_name)}.csv")

        plot_timeseries(metric_name, plot_series, output_dir, args.export_formats)

    cot_plot_series: list[dict[str, Any]] = []
    cot_stats_tables: list[pd.DataFrame] = []

    for label, data in series_data.items():
        run_entries: list[tuple[str, pd.DataFrame]] = []

        for run_name in data.selected_json_run_names:
            json_run = data.json_runs.get(run_name)
            if json_run is None or json_run.cot_df is None or json_run.cot_df.empty:
                continue
            run_entries.append((run_name, json_run.cot_df))

        stats = aggregate_cot_data(run_entries)

        cot_plot_series.append(
            {
                "label": label,
                "stats": stats,
            }
        )

        if not stats.empty:
            stats_to_save = stats.copy()
            stats_to_save.insert(0, "label", label)
            stats_to_save.insert(1, "env_name", data.env_name)
            cot_stats_tables.append(stats_to_save)

    if cot_stats_tables:
        cot_stats_df = pd.concat(cot_stats_tables, ignore_index=True)
        save_dataframe(cot_stats_df, output_dir / "cot_sweep_stats.csv")

    plot_cot_comparison(cot_plot_series, output_dir, args.export_formats, args.plot_baseline)

    logging.info("Finished. Outputs written to: %s", output_dir)


if __name__ == "__main__":
    main()