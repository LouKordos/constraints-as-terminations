import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scienceplots
import wandb

# ======================================================================================
# 1. SETUP & CONFIGURATION
# ======================================================================================

def setup_logging(log_level: str, output_path: Path):
    """Configures logging to both console and a file in the output directory."""
    log_level_upper = log_level.upper()
    logging.basicConfig(
        level=log_level_upper,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.FileHandler(output_path / "analysis.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Silence verbose logging from the wandb library
    logging.getLogger("wandb").setLevel(logging.WARNING)
    logging.info(f"Logging initialized at level: {log_level_upper}")

def setup_plotting_style():
    """Applies a consistent, professional plotting style for all generated figures."""
    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({
        'font.size': 28, 'axes.labelsize': 28, 'axes.titlesize': 28,
        'xtick.labelsize': 28, 'ytick.labelsize': 28, 'legend.fontsize': 28,
        'figure.titlesize': 28, 'figure.constrained_layout.use': True,
    })

def create_output_directory(base_name: str = "analysis") -> Path:
    """Creates a unique, timestamped directory for storing script outputs."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = f"{base_name}_{timestamp}"
    output_path = Path(dir_name)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created output directory: '{output_path}'") # Initial print before logging is set up
    return output_path

# ======================================================================================
# 2. DATA FETCHING & PARSING (WANDB & LOCAL)
# ======================================================================================

def parse_run_name_to_datetime(run_name: str) -> Optional[datetime]:
    """Converts a run name in 'YYYY-MM-DD-HH-M-SS' format to a datetime object."""
    try:
        return datetime.strptime(run_name, '%Y-%m-%d-%H-%M-%S')
    except ValueError:
        return None

def fetch_wandb_data(
    project_name: str,
    start_dt: datetime,
    end_dt: datetime,
    all_metrics: List[str],
    min_iterations: int
) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    """
    Fetches data from WandB for a given project and time range for all specified metrics.
    """
    api = wandb.Api()
    try:
        logging.debug(f"Accessing WandB project: '{project_name}'")
        runs = api.runs(project_name)
    except wandb.errors.CommError as e:
        logging.error(f"Could not access WandB project '{project_name}': {e}")
        return [], []

    run_dfs, summary_stats_list = [], []
    required_keys = ['_step'] + all_metrics

    # --- ENHANCED LOGGING: Added progress tracking ---
    total_runs = len(runs)
    processed_count = 0
    for run in runs:
        run_dt = parse_run_name_to_datetime(run.name)
        if not (run_dt and start_dt <= run_dt <= end_dt):
            continue
        
        processed_count += 1
        logging.info(f"  -> Fetching WandB data for run {processed_count}: {run.name}")
        
        history_df = pd.DataFrame(run.scan_history(keys=required_keys))
        if len(history_df) < min_iterations:
            logging.warning(f"    Skipping WandB run '{run.name}': Too few iterations ({len(history_df)} < {min_iterations}).")
            continue
        
        history_df.rename(columns={'_step': 'iteration'}, inplace=True)
        run_dfs.append(history_df)

        run_summary = {'run_name': run.name}
        for metric in all_metrics:
            if metric in history_df.columns:
                series = history_df[metric].dropna()
                if not series.empty:
                    last_10_percent_idx = int(len(series) * 0.9)
                    run_summary[metric] = series.iloc[last_10_percent_idx:].mean()
        
        if len(run_summary) > 1:
            summary_stats_list.append(run_summary)

    logging.info(f"Found {len(run_dfs)} valid WandB runs for analysis.")
    return run_dfs, summary_stats_list


def find_and_parse_json_data(
    base_dir: Path,
    start_dt: datetime,
    end_dt: datetime
) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    """Finds and parses metrics_summary.json files from local directories."""
    plot_dfs, summary_stats_list = [], []
    run_dir_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$')
    
    # --- ENHANCED LOGGING: Added progress tracking ---
    all_dirs = sorted(base_dir.iterdir())
    for i, run_dir in enumerate(all_dirs):
        if not (run_dir.is_dir() and run_dir_pattern.match(run_dir.name)):
            continue
        
        dir_dt = parse_run_name_to_datetime(run_dir.name)
        if not (dir_dt and start_dt <= dir_dt <= end_dt):
            continue

        logging.info(f"  -> Parsing local data for run {i+1}/{len(all_dirs)}: {run_dir.name}")
        
        eval_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('eval_checkpoint_')]
        if not eval_dirs:
            continue

        latest_eval_dir = max(eval_dirs, key=lambda d: int(re.search(r'(\d+)', d.name).group(1)))
        json_path = latest_eval_dir / 'metrics_summary.json'

        if not json_path.exists():
            logging.warning(f"    'metrics_summary.json' not found in '{latest_eval_dir}'. Skipping.")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            scenarios = data.get('fixed_command_scenarios_metrics', {})
            cot_records = []
            pattern = re.compile(r'cot_sweep_walk_x_flat_terrain_(\d+\.?\d*)')
            for key, metrics in scenarios.items():
                match = pattern.match(key)
                if match and isinstance(metrics, dict) and 'mean_cost_of_transport' in metrics:
                    velocity = float(match.group(1))
                    cot = metrics['mean_cost_of_transport']
                    cot_records.append({'velocity': velocity, 'cot': cot})
            if cot_records:
                plot_dfs.append(pd.DataFrame(cot_records))

            stats = {'run_name': run_dir.name}
            stats['rms_error_x'] = data.get('base_linear_velocity_x_rms_error', np.nan)
            stats['rms_error_y'] = data.get('base_linear_velocity_y_rms_error', np.nan)
            
            violations = data.get('constraint_violations_percent', {})
            if isinstance(violations, dict):
                torque_v = violations.get('joint_torque', {})
                accel_v = violations.get('joint_acceleration', {})
                stats['violation_torque'] = max(torque_v.values()) if isinstance(torque_v, dict) and torque_v else np.nan
                stats['violation_accel'] = max(accel_v.values()) if isinstance(accel_v, dict) and accel_v else np.nan
            summary_stats_list.append(stats)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.error(f"    Error parsing '{json_path}': {e}")

    logging.info(f"Found {len(plot_dfs)} valid local runs for CoT plot and {len(summary_stats_list)} for summary.")
    return plot_dfs, summary_stats_list

# ======================================================================================
# 3. DATA PROCESSING & AGGREGATION
# ======================================================================================

def align_timeseries_data(run_histories: List[pd.DataFrame], metric_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns time-series data from multiple runs and calculates mean and std error."""
    if not run_histories:
        return np.array([]), np.array([]), np.array([])
    
    aligned_dfs = []
    for i, df in enumerate(run_histories):
        df_processed = df.drop_duplicates(subset='iteration').set_index('iteration')
        df_processed.rename(columns={metric_name: f'run_{i}'}, inplace=True)
        aligned_dfs.append(df_processed)

    combined_df = pd.concat(aligned_dfs, axis=1, join='outer').sort_index()
    
    mean_values = combined_df.mean(axis=1)
    std_dev_values = combined_df.std(axis=1)
    run_counts = combined_df.count(axis=1)
    std_err_values = std_dev_values / np.sqrt(run_counts)

    stats_df = pd.DataFrame({'mean': mean_values, 'std_err': std_err_values}).dropna(subset=['mean'])
    return stats_df.index.values, stats_df['mean'].values, stats_df['std_err'].values

def align_cot_data(run_dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregates CoT data from multiple runs by velocity."""
    if not run_dfs:
        return np.array([]), np.array([]), np.array([])
    
    combined_df = pd.concat(run_dfs, ignore_index=True)
    stats_df = combined_df.groupby('velocity')['cot'].agg(['mean', 'std', 'count']).reset_index()
    stats_df['std_err'] = stats_df['std'] / np.sqrt(stats_df['count'])
    stats_df.sort_values('velocity', inplace=True)
    return stats_df['velocity'].values, stats_df['mean'].values, stats_df['std_err'].values

def generate_summary_report(
    all_series_data: Dict[str, Dict[str, Any]],
    cot_velocity_range: List[float],
    output_path: Path
) -> None:
    """Calculates, logs, and saves summary statistics for all series."""
    report_lines = [f"Analysis Report - Generated on {datetime.now(tz=None).strftime('%Y-%m-%d %H:%M:%S')}\n"]
    logging.info("="*50 + "\n📋 SUMMARY STATISTICS\n" + "="*50)

    min_vel, max_vel = cot_velocity_range
    metrics_map = {
        'Curriculum/terrain_levels': 'Final Terrain Level',
        'Episode/MaxJointPos': 'Final Max Joint Pos',
        'Episode/MaxAirTime': 'Final Max Air Time',
        'Episode/EnergyConsumed': 'Final Energy Consumed',
        'mean_cot_filtered': f'Mean CoT (Velocities {min_vel}-{max_vel})',
        'violation_torque': 'Max Constraint Violation (Torque %)',
        'violation_accel': 'Max Constraint Violation (Accel %)',
        'rms_error_xy_mean': 'Mean Base Velocity RMS Error (X,Y)'
    }

    for label, data in all_series_data.items():
        series_header = f"\n--- Series: '{label}' ---"
        logging.info(series_header)
        report_lines.append(series_header)

        wandb_df = pd.DataFrame(data.get('wandb_summaries', []))
        json_df = pd.DataFrame(data.get('json_summaries', []))
        
        if not json_df.empty:
            json_df['rms_error_xy_mean'] = json_df[['rms_error_x', 'rms_error_y']].mean(axis=1)
            full_cot_df = pd.concat(data.get('cot_dfs', []), ignore_index=True)
            if not full_cot_df.empty:
                filtered_cot = full_cot_df[full_cot_df['velocity'].between(min_vel, max_vel)]['cot']
                if not filtered_cot.empty:
                    # Use a new column to avoid potential SettingWithCopyWarning
                    json_df_copy = json_df.copy()
                    json_df_copy.loc[0, 'mean_cot_filtered'] = filtered_cot.mean()
                    json_df = json_df_copy


        stats_df = pd.concat([wandb_df, json_df], axis=1)

        for key, name in metrics_map.items():
            if key not in stats_df.columns:
                continue
            
            data_series = stats_df[key].dropna()
            n = len(data_series)
            
            if n == 0:
                line = f"  {name:<45}: Not available"
            else:
                mean_val = data_series.mean()
                if n > 1:
                    std_err = data_series.std() / np.sqrt(n)
                    margin_of_error = 1.96 * std_err
                    line = f"  {name:<45}: {mean_val:.4f} ± {margin_of_error:.4f}  (95% CI, n={n})"
                else:
                    line = f"  {name:<45}: {mean_val:.4f}  (n=1, CI not applicable)"
            
            logging.info(line)
            report_lines.append(line)

    summary_file = output_path / "summary_statistics.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(report_lines))
    logging.info(f"\nSummary report saved to '{summary_file}'")

# ======================================================================================
# 4. PLOTTING
# ======================================================================================

def plot_timeseries(plot_data: List[Dict], metric_name: str, output_path: Path):
    """Generates and saves a time-series plot."""
    logging.info(f"Generating time-series plot for '{metric_name}'...")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    linewidth = 2.0 if metric_name == "Curriculum/terrain_levels" else 1.5
    label_map = {
        "Curriculum/terrain_levels": "Terrain Level",
        "Episode/MaxJointPos": "Max. Joint Position (rad)",
        "Episode/MaxAirTime": "Max. Air Time (s)",
        "Episode/EnergyConsumed": "Energy Consumed (J)"
    }
    ylabel = label_map.get(metric_name, metric_name.split('/')[-1].replace('_', ' ').title())

    for i, data in enumerate(plot_data):
        if data['iterations'].size == 0:
            logging.warning(f"No data to plot for series '{data['label']}' for metric '{metric_name}'. Skipping.")
            continue
        
        ax.plot(data['iterations'], data['means'], label=data['label'], color=colors[i], linewidth=linewidth)
        if not np.all(np.isnan(data['std_errs'])):
            ci_lower = data['means'] - 1.96 * data['std_errs']
            ci_upper = data['means'] + 1.96 * data['std_errs']
            ax.fill_between(data['iterations'], ci_lower, ci_upper, color=colors[i], alpha=0.2)
        else:
             logging.debug(f"Series '{data['label']}' has only one run. Plotting line only.")

    title = f'{ylabel} Progression'
    ax.set_title(title, fontsize=34)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.legend()
    
    if metric_name == "Episode/EnergyConsumed":
        ax.set_ylim(0, 2000)
    
    filename = output_path / f"plot_wandb_{re.sub(r'[/]', '_', metric_name)}.pdf"
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Time-series plot saved to '{filename}'")

def plot_cot_comparison(plot_data: List[Dict], plot_baseline: bool, output_path: Path):
    """Generates and saves the CoT velocity sweep comparison plot."""
    logging.info("Generating CoT velocity sweep plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, series_data in enumerate(plot_data):
        if series_data['velocities'].size == 0:
            logging.warning(f"No CoT data to plot for series '{series_data['label']}'. Skipping.")
            continue
        
        ax.plot(series_data['velocities'], series_data['means'], label=series_data['label'], 
                color=colors[i], linestyle='-', marker='o', linewidth=2.5, markersize=8)
        
        ci_lower = series_data['means'] - 1.96 * series_data['std_errs']
        ci_upper = series_data['means'] + 1.96 * series_data['std_errs']
        ax.fill_between(series_data['velocities'], ci_lower, ci_upper, color=colors[i], alpha=0.2)
        
    if plot_baseline:
        logging.debug("Overlaying baseline data on CoT plot.")
        baseline_v = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        baseline_c = [0.92, 0.75, 0.65, 0.6, 0.55, 0.5]
        ax.plot(baseline_v, baseline_c, label='Baseline (Ref. Paper)', color='dimgray', 
                linestyle='--', marker='s', linewidth=2.5, markersize=8)

    ax.set_title('Cost of Transport Velocity Sweep', fontsize=34)
    ax.set_xlabel('Commanded Velocity (m/s)')
    ax.set_ylabel('Cost of Transport')
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10, prune='both'))
    
    filename = output_path / "plot_cot_sweep.pdf"
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"CoT plot saved to '{filename}'")

# ======================================================================================
# 5. MAIN EXECUTION
# ======================================================================================

def parse_flexible_datetime(dt_str: str, is_end_date: bool = False) -> datetime:
    """Parses a string that could be a date or a datetime."""
    try:
        return datetime.strptime(dt_str, '%Y-%m-%d-%H-%M-%S')
    except ValueError:
        dt_obj = datetime.strptime(dt_str, '%Y-%m-%d')
        return dt_obj.replace(hour=23, minute=59, second=59) if is_end_date else dt_obj

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis script for RL experiments from WandB and local JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Series definition
    parser.add_argument("--labels", type=str, nargs='+', required=True, help="Labels for each experiment series.")
    parser.add_argument("--projects", type=str, nargs='+', required=True, help="WandB projects for each series: 'entity/project'.")
    parser.add_argument("--start_times", type=str, nargs='+', required=True, help="Start date/datetime for each series: 'YYYY-MM-DD' or 'YYYY-MM-DD-HH-MM-SS'.")
    parser.add_argument("--end_times", type=str, nargs='+', required=True, help="End date/datetime for each series.")
    
    # Data source and metrics
    parser.add_argument("--base_dirs", type=str, nargs='+', required=True, help="Base directory for local files for each series.")
    parser.add_argument("--wandb_plot_metric", type=str, default="Curriculum/terrain_levels", help="Primary WandB metric for a time-series plot.")
    parser.add_argument("--min_iterations", type=int, default=500, help="Minimum data points for a wandb run to be included.")
    parser.add_argument("--cot_velocity_range", type=float, nargs=2, default=[0.6, 1.6], help="Min and max velocity for CoT summary statistic and plot.")
    
    # Output and plotting
    parser.add_argument("--output_name", type=str, default="analysis_run", help="Base name for the output directory.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="Set the logging level.")
    parser.add_argument("--no_baseline", action='store_true', help="If set, disables plotting the hardcoded baseline on the CoT plot.")

    args = parser.parse_args()
    num_series = len(args.labels)
    if not (num_series == len(args.projects) == len(args.start_times) == len(args.end_times) == len(args.base_dirs)):
        parser.error("The number of --labels, --projects, --start_times, --end_times, and --base_dirs must be the same.")

    for i in range(num_series):
        try:
            start_dt = parse_flexible_datetime(args.start_times[i])
            end_dt = parse_flexible_datetime(args.end_times[i], is_end_date=True)
            if start_dt > end_dt:
                print(f"⚠️ WARNING: For series '{args.labels[i]}', start time {start_dt} is after end time {end_dt}.")
        except ValueError:
            parser.error(f"Invalid date format for series '{args.labels[i]}'. Use YYYY-MM-DD or YYYY-MM-DD-HH-MM-SS.")

    output_dir = create_output_directory(args.output_name)
    setup_logging(args.log_level, output_dir)
    logging.info("Script execution started with the following arguments:")
    for arg, value in sorted(vars(args).items()): logging.info(f"  --{arg}: {value}")

    setup_plotting_style()
    wandb_summary_metrics = ["Curriculum/terrain_levels", "Episode/MaxJointPos", "Episode/MaxAirTime", "Episode/EnergyConsumed"]
    all_metrics_to_plot = sorted(list(set([args.wandb_plot_metric] + wandb_summary_metrics)))
    all_series_data = {}

    for i in range(num_series):
        label = args.labels[i]
        project = args.projects[i]
        start_dt = parse_flexible_datetime(args.start_times[i])
        end_dt = parse_flexible_datetime(args.end_times[i], is_end_date=True)
        base_path = Path(args.base_dirs[i])

        logging.info(f"\n{'='*20} Processing Series {i+1}/{num_series}: '{label}' {'='*20}")
        logging.info(f"Time range: {start_dt} to {end_dt}")
        logging.info(f"Local data path: {base_path}")

        if not base_path.is_dir():
            logging.error(f"The specified base directory for series '{label}' does not exist: {base_path}")
            sys.exit(1)
        
        wandb_run_dfs, wandb_summaries = fetch_wandb_data(project, start_dt, end_dt, all_metrics_to_plot, args.min_iterations)
        cot_dfs, json_summaries = find_and_parse_json_data(base_path, start_dt, end_dt)
        
        all_series_data[label] = {
            "wandb_run_dfs": wandb_run_dfs,
            "wandb_summaries": wandb_summaries,
            "json_summaries": json_summaries,
            "cot_dfs": cot_dfs
        }

    # --- Generate Outputs ---
    generate_summary_report(all_series_data, args.cot_velocity_range, output_dir)
    
    for metric in all_metrics_to_plot:
        plot_data_for_metric = []
        for label, series_data in all_series_data.items():
            metric_dfs = [df[['iteration', metric]].copy().dropna() for df in series_data['wandb_run_dfs'] if metric in df.columns]
            iters, means, stderrs = align_timeseries_data(metric_dfs, metric)
            plot_data_for_metric.append({'label': label, 'iterations': iters, 'means': means, 'std_errs': stderrs})
        plot_timeseries(plot_data_for_metric, metric, output_dir)

    cot_plot_data = []
    min_vel, max_vel = args.cot_velocity_range
    for label, series_data in all_series_data.items():
        filtered_dfs = [df[df['velocity'].between(min_vel, max_vel)] for df in series_data['cot_dfs']]
        vels, means, stderrs = align_cot_data([df for df in filtered_dfs if not df.empty])
        cot_plot_data.append({'label': label, 'velocities': vels, 'means': means, 'std_errs': stderrs})
    plot_cot_comparison(cot_plot_data, not args.no_baseline, output_dir)

    logging.info("✅ Analysis complete.")

if __name__ == "__main__":
    main()
