"""
Compares the swing height distributions of two different time step ranges
from a single simulation data file.

This script takes one .npz data file and two time ranges as input. It computes 
the swing heights for each range using the 'metrics_utils' library and then 
generates two plots:
1. A side-by-side box plot.
2. An overlaid, semi-transparent histogram.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from typing import List, Dict

# --- Import helper functions from your library ---
try:
    from metrics_utils import compute_swing_heights, compute_trimmed_histogram_data
except ImportError:
    print("❌ Error: Could not import from 'metrics_utils'.")
    print("Please ensure 'compare_swing_heights.py' is in the same directory as 'metrics_utils.py' or that the library is in your PYTHONPATH.")
    exit(1)


# --- Matplotlib Styling (copied for consistency) ---
plt.style.use(['science', 'ieee'])
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 22,
    'figure.constrained_layout.use': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})

# --- Main Script Logic ---

def process_range(full_data: Dict[str, np.ndarray], start_step: int, end_step: int) -> List[float]:
    """Slices data, computes swing heights for the slice, and returns a flat list."""
    print(f"Processing time steps from {start_step} to {end_step}...")

    time_slice = slice(start_step, end_step + 1)
    
    sliced_data = {
        k: (v[time_slice] if (k.endswith("_array") or k == "sim_times") else v) 
        for k, v in full_data.items()
    }

    sim_times = sliced_data['sim_times']
    if len(sim_times) < 2:
        print(f"Warning: The range {start_step}-{end_step} contains fewer than 2 data points. Skipping.")
        return []

    contact_state_array = sliced_data['contact_state_array']
    foot_positions_contact_frame = sliced_data['foot_positions_contact_frame_array']
    foot_labels = list(sliced_data['foot_labels'])

    t0, t1 = float(sim_times[0]), float(sim_times[-1])
    full_resets = full_data["reset_times"]
    resets_in_window = full_resets[(full_resets >= t0) & (full_resets <= t1)]
    step_dt = sim_times[1] - sim_times[0]
    reset_timesteps = [int(round((rt - t0) / step_dt)) for rt in resets_in_window]
    
    foot_heights_contact_frame = foot_positions_contact_frame[:, :, 2]

    swing_heights_dict = compute_swing_heights(
        contact_state=contact_state_array,
        foot_heights_contact=foot_heights_contact_frame,
        reset_steps=reset_timesteps,
        foot_labels=foot_labels
    )

    all_swing_heights = [height for foot_heights in swing_heights_dict.values() for height in foot_heights]
    print(f"Found {len(all_swing_heights)} swing events in this range.")
    return all_swing_heights

def main():
    """Main function to parse arguments and generate the plots."""
    parser = argparse.ArgumentParser(
        description="Compare swing height distributions between two time ranges in a single data file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Arguments ---
    parser.add_argument("--data_file", type=str, required=True, help="Path to the .npz data file.")
    parser.add_argument("--range1_start", type=int, required=True, help="Start time step for the first range.")
    parser.add_argument("--range1_end", type=int, required=True, help="End time step (inclusive) for the first range.")
    parser.add_argument("--range1_label", type=str, default="Range 1", help="Label for the first range on the plot.")
    parser.add_argument("--range2_start", type=int, required=True, help="Start time step for the second range.")
    parser.add_argument("--range2_end", type=int, required=True, help="End time step (inclusive) for the second range.")
    parser.add_argument("--range2_label", type=str, default="Range 2", help="Label for the second range on the plot.")
    parser.add_argument("--output_file", type=str, default="swing_height_comparison.pdf", help="Path to save the output box plot. The histogram will be saved with a '_histogram' suffix.")
    parser.add_argument("--show", action="store_true", help="Display the plots interactively after saving.")
    
    args = parser.parse_args()

    # --- Data Loading and Processing ---
    try:
        full_data = dict(np.load(args.data_file, allow_pickle=True))
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {args.data_file}")
        return

    heights_range1 = process_range(full_data, args.range1_start, args.range1_end)
    heights_range2 = process_range(full_data, args.range2_start, args.range2_end)

    if not heights_range1 or not heights_range2:
        print("\nCould not generate plots because at least one range had no valid swing data.")
        return

    # --- 1. Box Plot ---
    fig_box, ax_box = plt.subplots(figsize=(12, 6))
    ax_box.boxplot([heights_range1, heights_range2], labels=[args.range1_label, args.range2_label], showmeans=True, showfliers=False)
    ax_box.set_title("Swing Height Distribution Comparison")
    ax_box.set_ylabel(r'Swing Height ($\text{m}$)')
    #ax_box.set_xlabel("Time Step Range")
    #ax_box.grid(False, linestyle='--', alpha=0.6)

    # --- 2. Histogram Plot ---
    fig_hist, ax_hist = plt.subplots(figsize=(12, 8))
    
    # Prepare data using the trimmed histogram utility
    counts1, edges1 = compute_trimmed_histogram_data(np.array(heights_range1))
    counts2, edges2 = compute_trimmed_histogram_data(np.array(heights_range2))
    
    # Plot using ax.stairs for the correct style, with transparency
    ax_hist.stairs(counts1, edges1, fill=True, alpha=0.7, label=args.range1_label)
    ax_hist.stairs(counts2, edges2, fill=True, alpha=0.7, label=args.range2_label)
    
    ax_hist.set_title("Swing Height Distribution Comparison (Histogram)")
    ax_hist.set_xlabel(r'Max Swing Height ($\text{m}$)')
    ax_hist.set_ylabel("Frequency (Count)")
    ax_hist.legend(loc='upper right')
    #ax_hist.grid(False, linestyle='--', alpha=0.6)

    # --- Saving Plots ---
    # Define histogram output path based on the main output file
    base, ext = os.path.splitext(args.output_file)
    hist_output_file = f"{base}_histogram{ext}"

    try:
        fig_box.savefig(args.output_file, dpi=600, bbox_inches='tight')
        print(f"\n✅ Box plot successfully saved to: {args.output_file}")
        fig_hist.savefig(hist_output_file, dpi=600, bbox_inches='tight')
        print(f"✅ Histogram successfully saved to: {hist_output_file}")
    except Exception as e:
        print(f"\n❌ Failed to save plots. Error: {e}")
        
    if args.show:
        print("Displaying plots...")
        plt.show()

if __name__ == "__main__":
    main()
