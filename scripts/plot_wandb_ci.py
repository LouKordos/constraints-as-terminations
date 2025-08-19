import argparse
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import scienceplots
import re

# Apply a professional plotting style
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

def parse_run_name_to_datetime(run_name: str) -> datetime | None:
    """Converts a run name in 'YYYY-MM-DD-HH-MM-SS' format to a datetime object."""
    try:
        return datetime.strptime(run_name, '%Y-%m-%d-%H-%M-%S')
    except ValueError:
        return None

def fetch_runs_data(project_name: str, start_time_str: str, end_time_str: str, metric_name: str, min_iterations: int) -> list[pd.DataFrame]:
    """Fetches complete historical data for a metric from runs within a time range, filtering by length."""
    api = wandb.Api()
    try:
        print(f"   -> Accessing project: '{project_name}'")
        runs = api.runs(project_name)
    except wandb.errors.CommError as e:
        print(f"❌ Error: Could not find project '{project_name}'. Please check the name and your permissions.")
        print(f"   Details: {e}")
        return []

    start_dt = parse_run_name_to_datetime(start_time_str)
    end_dt = parse_run_name_to_datetime(end_time_str)
    if not start_dt or not end_dt:
        raise ValueError("Invalid start or end time format. Expected 'YYYY-MM-DD-HH-MM-SS'.")

    print(f"   -> Filtering runs from {start_dt} to {end_dt}...")
    run_histories = []

    for run in runs:
        run_dt = parse_run_name_to_datetime(run.name)
        if run_dt and start_dt <= run_dt <= end_dt:
            history_scan = run.scan_history(keys=['_step', metric_name])
            history_df = pd.DataFrame(history_scan)

            if history_df.empty or metric_name not in history_df.columns:
                continue
            if len(history_df) < min_iterations:
                print(f"   ⚠️ Warning: Run '{run.name}' has {len(history_df)} data points (< {min_iterations}). Skipping.")
                continue

            history_df.dropna(subset=[metric_name], inplace=True)
            history_df.rename(columns={'_step': 'iteration'}, inplace=True)
            if not history_df.empty:
                run_histories.append(history_df[['iteration', metric_name]])
    
    print(f"📊 Found {len(run_histories)} valid runs for this series.")
    return run_histories

def process_and_align_data(run_histories: list[pd.DataFrame], metric_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns data from multiple runs and calculates mean and standard error."""
    if not run_histories:
        return np.array([]), np.array([]), np.array([])

    aligned_dfs = []
    for i, df in enumerate(run_histories):
        if df['iteration'].duplicated().any():
            df = df.groupby('iteration', as_index=False).mean()
        df.set_index('iteration', inplace=True)
        df.rename(columns={metric_name: f'run_{i}'}, inplace=True)
        aligned_dfs.append(df)

    combined_df = pd.concat(aligned_dfs, axis=1, join='outer')
    combined_df.sort_index(inplace=True)

    mean_values = combined_df.mean(axis=1)
    std_dev_values = combined_df.std(axis=1)
    run_counts = combined_df.count(axis=1)
    std_err_values = std_dev_values / np.sqrt(run_counts.where(run_counts > 0, np.nan))

    stats_df = pd.DataFrame({'mean': mean_values, 'std_err': std_err_values})
    # This line is now safe, but we will handle NaNs in the plotting function
    stats_df.dropna(subset=['mean'], inplace=True)

    return stats_df.index.values, stats_df['mean'].values, stats_df['std_err'].values

def plot_comparison(plot_data: list, metric_name: str, output_file: str):
    """Generates and saves a comparison plot with multiple confidence intervals."""
    print(f"\n📈 Generating comparison plot for '{metric_name}'...")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, data in enumerate(plot_data):
        label = data['label']
        iterations = data['iterations']
        means = data['means']
        std_errs = data['std_errs']
        color = colors[i % len(colors)]

        if iterations.size == 0:
            print(f"⚠️ No data available to plot for series '{label}'. Skipping.")
            continue

        # --- CRITICAL FIX: Handle the single-run case gracefully ---
        # Check if std_errs is entirely NaN
        is_single_run = np.all(np.isnan(std_errs))

        if is_single_run:
            # If it's a single run, just plot the line.
            print(f"   -> Series '{label}' has only one run. Plotting line without confidence interval.")
            ax.plot(iterations, means, label=label, color=color, linewidth=2)
        else:
            # If there are multiple runs, plot the mean and the confidence interval.
            ax.plot(iterations, means, label=label, color=color, linewidth=2)
            ci_lower = means - 1.96 * std_errs
            ci_upper = means + 1.96 * std_errs
            ax.fill_between(iterations, ci_lower, ci_upper, color=color, alpha=0.2)
        # --- END FIX ---


    ax.set_title(f'{metric_name} Comparison',fontsize=34)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric_name.split('/')[-1].replace('_', ' ').title())
    ax.legend()
    
    print(f"💾 Saving plot to '{output_file}'...")
    fig.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print("✅ Plot saved successfully.")

def sanitize_filename(name: str) -> str:
    """Removes characters that are problematic for filenames."""
    return re.sub(r'[/\\?%*:|"<>]', '_', name)

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Fetch and compare multiple WandB experiment series on a single plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--projects", type=str, nargs='+', required=True, help="WandB project(s): 'entity/project'. One per series.")
    parser.add_argument("--labels", type=str, nargs='+', required=True, help="A label for each experiment series (e.g., 'Baseline' 'New Method').")
    parser.add_argument("--start_times", type=str, nargs='+', required=True, help="Start run name for each series: 'YYYY-MM-DD-HH-MM-SS'.")
    parser.add_argument("--end_times", type=str, nargs='+', required=True, help="End run name for each series: 'YYYY-MM-DD-HH-MM-SS'.")
    
    parser.add_argument("--metric", type=str, default="Curriculum/terrain_levels", help="Metric to plot.")
    parser.add_argument("--min_iterations", type=int, default=500, help="Minimum data points for a run to be included.")
    parser.add_argument("--output_file", type=str, default="wandb_comparison_plot.pdf", help="Path for the output plot file.")
    args = parser.parse_args()

    if not (len(args.projects) == len(args.labels) == len(args.start_times) == len(args.end_times)):
        raise ValueError("Error: The number of --projects, --labels, --start_times, and --end_times must be the same.")

    plot_data = []
    num_series = len(args.labels)

    for i in range(num_series):
        project = args.projects[i]
        label = args.labels[i]
        start_time = args.start_times[i]
        end_time = args.end_times[i]
        
        print(f"\n--- Processing Series {i+1}/{num_series}: '{label}' ---")
        
        run_histories = fetch_runs_data(project, start_time, end_time, args.metric, args.min_iterations)
        iterations, means, std_errs = process_and_align_data(run_histories, args.metric)
        
        plot_data.append({
            'label': label,
            'iterations': iterations,
            'means': means,
            'std_errs': std_errs,
        })

    plot_comparison(plot_data, args.metric, args.output_file)

if __name__ == "__main__":
    main()
