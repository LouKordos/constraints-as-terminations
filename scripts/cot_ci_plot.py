import argparse
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # <<< IMPORTED for y-tick resolution
from pathlib import Path
from datetime import datetime
from typing import Optional
import scienceplots

# Apply the professional plotting style
plt.style.use(['science', 'ieee'])
plt.rcParams.update({
    'font.size': 28,
    'axes.labelsize': 28,
    'axes.titlesize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 28,
    'figure.titlesize': 28,
    'figure.constrained_layout.use': True,
})

def parse_single_run_data(json_path: Path) -> Optional[pd.DataFrame]:
    """
    Parses a single metrics_summary.json file and prints the extracted data.
    """
    print(f"      - Attempting to parse {json_path.name}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        scenarios_dict = data.get('fixed_command_scenarios_metrics', {})
        if not scenarios_dict:
            print(f"      - ⚠️ Warning: Key 'fixed_command_scenarios_metrics' not found. Skipping file.")
            return None
        records = []
        scenario_pattern = re.compile(r'cot_sweep_walk_x_flat_terrain_(\d+\.?\d*)')
        for key, metrics in scenarios_dict.items():
            match = scenario_pattern.match(key)
            if match:
                try:
                    velocity_str = match.group(1)
                    velocity = float(velocity_str)
                    if not isinstance(metrics, dict):
                        continue
                    cot = metrics.get('mean_cost_of_transport')
                    if cot is None:
                        continue
                    records.append({'velocity': velocity, 'cot': cot})
                except (ValueError, TypeError):
                    continue
        if not records:
            print(f"      - ⚠️ Warning: No valid CoT data points found in file.")
            return None
        
        df = pd.DataFrame(records)
        print(f"      - ✅ Successfully parsed {len(df)} data points:")
        print(df.to_string(index=False).replace('\n', '\n      - '))
        return df
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"      - ❌ Error reading or parsing file: {e}")
        return None

def find_and_parse_runs(base_dir: Path, start_dt: Optional[datetime], end_dt: Optional[datetime]) -> list[pd.DataFrame]:
    """
    Finds valid runs within a datetime range and parses their JSON data.
    """
    run_dataframes = []
    run_dir_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$')
    for run_dir in sorted(base_dir.iterdir()):
        if run_dir.is_dir() and run_dir_pattern.match(run_dir.name):
            try:
                dir_dt = datetime.strptime(run_dir.name, '%Y-%m-%d-%H-%M-%S')
                if (start_dt and dir_dt < start_dt) or (end_dt and dir_dt > end_dt):
                    print(f"   -> Skipping directory {run_dir.name} (outside datetime range).")
                    continue
            except ValueError:
                continue

            print(f"   -> Processing directory: {run_dir.name}")
            eval_dirs = {}
            eval_pattern = re.compile(r'eval_checkpoint_(\d+)_seed_(\d+)')
            for eval_dir in run_dir.iterdir():
                if eval_dir.is_dir():
                    match = eval_pattern.match(eval_dir.name)
                    if match:
                        checkpoint_num = int(match.group(1))
                        eval_dirs[checkpoint_num] = eval_dir
            if not eval_dirs:
                print(f"      - ⚠️ Warning: No 'eval_checkpoint_*' directories found. Skipping run.")
                continue
                
            latest_checkpoint = max(eval_dirs.keys())
            latest_eval_dir = eval_dirs[latest_checkpoint]
            print(f"      - Found latest checkpoint: {latest_checkpoint} in '{latest_eval_dir.name}'")
            
            json_file_path = latest_eval_dir / 'metrics_summary.json'
            if json_file_path.exists():
                df = parse_single_run_data(json_file_path)
                if df is not None:
                    run_dataframes.append(df)
            else:
                print(f"      - ⚠️ Warning: 'metrics_summary.json' not found. Skipping run.")
    return run_dataframes

def process_and_align_data(run_dataframes: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not run_dataframes:
        return np.array([]), np.array([]), np.array([])
    combined_df = pd.concat(run_dataframes, ignore_index=True)
    stats_df = combined_df.groupby('velocity')['cot'].agg(['mean', 'std', 'count']).reset_index()
    stats_df['std_err'] = stats_df['std'] / np.sqrt(stats_df['count'])
    stats_df.sort_values('velocity', inplace=True)
    return stats_df['velocity'].values, stats_df['mean'].values, stats_df['std_err'].values

def plot_comparison(plot_data: list, output_file: str, plot_baseline: bool):
    print(f"\n📈 Generating comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, series_data in enumerate(plot_data):
        label, velocities, means, std_errs = series_data.values()
        color = colors[i % len(colors)]
        if velocities.size == 0:
            print(f"   ⚠️ Warning: No data to plot for series '{label}'. Skipping.")
            continue
        print(f"   -> Plotting series: '{label}'")
        ax.plot(velocities, means, label=label, color=color, linestyle='-', linewidth=2.5, marker='o', markersize=8)
        ci_lower = means - 1.96 * std_errs
        ci_upper = means + 1.96 * std_errs
        ax.fill_between(velocities, ci_lower, ci_upper, color=color, alpha=0.2)
    if plot_baseline:
        print("   -> Overlaying baseline data points.")
        baseline_velocities = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        baseline_cot = [0.92, 0.75, 0.65, 0.6, 0.55, 0.5]
        baseline_color_index = len(plot_data) % len(colors)
        ax.plot(baseline_velocities, baseline_cot, label='Baseline (Ref. Paper)', color=colors[baseline_color_index], linestyle='--', marker='s', linewidth=2.5, markersize=8)
    
    # <<< UPDATED: Changed title
    ax.set_title('Cost of Transport Velocity Sweep', fontsize=34)
    ax.set_xlabel('Commanded Velocity (m/s)')
    ax.set_ylabel('Cost of Transport')
    ax.legend()
    
    # <<< NEW: Increase the number of ticks on the y-axis
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10, prune='both'))

    print(f"💾 Saving plot to '{output_file}'...")
    fig.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print("✅ Plot saved successfully.")

def parse_flexible_datetime(dt_str: str, is_end_date: bool = False) -> datetime:
    """Parses a string that could be a date or a datetime."""
    try:
        return datetime.strptime(dt_str, '%Y-%m-%d-%H-%M-%S')
    except ValueError:
        dt_obj = datetime.strptime(dt_str, '%Y-%m-%d')
        if is_end_date:
            return dt_obj.replace(hour=23, minute=59, second=59)
        return dt_obj

def main():
    parser = argparse.ArgumentParser(description="Generate a CI plot for Cost of Transport from local JSON files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("base_dir", type=str, help="The base directory containing all training run folders.")
    parser.add_argument("--labels", type=str, nargs='+', required=True, help="Labels for each experiment series (e.g., 'V1' 'V2').")
    parser.add_argument("--start_dates", type=str, nargs='+', required=True, help="Start date/datetime for each series: 'YYYY-MM-DD' or 'YYYY-MM-DD-HH-MM-SS'.")
    parser.add_argument("--end_dates", type=str, nargs='+', required=True, help="End date/datetime for each series: 'YYYY-MM-DD' or 'YYYY-MM-DD-HH-MM-SS'.")
    parser.add_argument("--output_file", type=str, default="cot_comparison_plot.pdf", help="Path for the output plot file.")
    parser.add_argument("--plot_baseline", action='store_true', help="If set, plots a hardcoded baseline from a reference paper.")
    args = parser.parse_args()

    if not (len(args.labels) == len(args.start_dates) == len(args.end_dates)):
        raise ValueError("Error: The number of --labels, --start_dates, and --end_dates must be the same.")

    base_path = Path(args.base_dir)
    if not base_path.is_dir():
        print(f"❌ Error: The specified base directory does not exist: {args.base_dir}")
        return

    plot_data_list = []
    num_series = len(args.labels)
    for i in range(num_series):
        label = args.labels[i]
        start_date_str = args.start_dates[i]
        end_date_str = args.end_dates[i]
        
        print(f"\n--- Processing Series {i+1}/{num_series}: '{label}' ({start_date_str} to {end_date_str}) ---")
        
        try:
            start_dt = parse_flexible_datetime(start_date_str)
            end_dt = parse_flexible_datetime(end_date_str, is_end_date=True)
            print(f"   -> Parsed datetime range: {start_dt} to {end_dt}")
        except ValueError:
            print(f"❌ Error: Invalid date/datetime format for series '{label}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DD-HH-MM-SS'.")
            continue

        run_dfs = find_and_parse_runs(base_path, start_dt, end_dt)
        if not run_dfs:
            print(f"   ❌ No valid runs found for series '{label}' in the given datetime range.")
        else:
            print(f"   📊 Found {len(run_dfs)} valid runs to aggregate for series '{label}'.")

        velocities, means, std_errs = process_and_align_data(run_dfs)
        plot_data_list.append({'label': label, 'velocities': velocities, 'means': means, 'std_errs': std_errs})

    plot_comparison(plot_data_list, args.output_file, args.plot_baseline)

if __name__ == "__main__":
    main()
