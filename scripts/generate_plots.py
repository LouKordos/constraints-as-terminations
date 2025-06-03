"""
Takes numpy buffers stored on disk and generates interactive plots, saving them to disk using picke.dump to be able to open them after generation.
"""

import argparse
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import time

# Disable interactive display until --interactive is set
plt.ioff()

def load_data(npz_path, summary_path=None):
    """
    Load simulation data from a .npz file and optional metrics summary JSON.
    """
    data = np.load(npz_path, allow_pickle=True)
    metrics = None
    if summary_path and os.path.isfile(summary_path):
        with open(summary_path, 'r') as f:
            metrics = json.load(f)
    return data, metrics

def draw_limits(ax, term, bounds):
    """
    ax     : matplotlib Axes
    term   : either a joint name or a global constraint key
    bounds : the dict returned by load_constraint_bounds()
    """
    lb, ub = bounds.get(term, (None, None))

    if lb is None and ub is None:
        print(f"[warn] no bounds for {term}, skipping.")
        return

    handles, labels = ax.get_legend_handles_labels()

    # helper to draw one line if that bound exists
    def _draw(val: float, suffix: str):
        lbl = f"{term}_{suffix}={val:.3f}"
        if lbl not in labels:
            ax.axhline(val, linestyle='--', linewidth=1, color='red', label=lbl)
            labels.append(lbl)  # so we don’t dup next time
        else:
            ax.axhline(val, linestyle='--', linewidth=1, color='red')

    # upper bound
    if ub is not None:
        _draw(ub, 'ub')

    # lower bound
    if lb is not None:
        _draw(lb, 'lb')

def draw_resets(ax, reset_times):
    for reset_time in reset_times:
        ax.axvline(x=reset_time, linestyle=":", linewidth=1, color="orange", label='reset' if reset_time == reset_times[0] else None)

def create_height_map_animation(height_map_sequence: np.ndarray, foot_positions_sequence: np.ndarray, output_path: str, fps: int = 30, sensor=None):
    """
    Create and save an animation of the height map over time,
    reshaping the 1D ray output into the 2D grid based on the sensor's pattern_cfg.
    """
    if sensor is None:
        raise ValueError("RayCaster sensor instance must be provided to determine grid dimensions.")

    # Get the ray start positions for the first environment
    ray_starts = sensor.ray_starts[0].cpu().numpy()  # shape: (R, 3)
    x_coords = ray_starts[:, 0]
    y_coords = ray_starts[:, 1]
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    Nx = len(unique_x)
    Ny = len(unique_y)
    ordering = sensor.cfg.pattern_cfg.ordering

    fig, ax = plt.subplots()

    # Initial frame
    frame0 = height_map_sequence[0]
    if ordering == 'xy':
        grid0 = frame0.reshape((Ny, Nx))
    else:  # 'yx'
        grid0 = frame0.reshape((Nx, Ny)).T
    heatmap = ax.imshow(grid0, origin='lower')

    scatter = ax.scatter(
        foot_positions_sequence[0][:, 0],
        foot_positions_sequence[0][:, 1],
        c='red',
        s=20
    )
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Terrain Height')

    def animate_frame(frame_index):
        frame = height_map_sequence[frame_index]
        if ordering == 'xy':
            grid = frame.reshape((Ny, Nx))
        else:
            grid = frame.reshape((Nx, Ny)).T
        heatmap.set_data(grid)
        scatter.set_offsets(foot_positions_sequence[frame_index][:, :2])
        return heatmap, scatter

    animation_obj = animation.FuncAnimation(
        fig,
        animate_frame,
        frames=len(height_map_sequence),
        blit=True
    )
    animation_obj.save(output_path, fps=fps)
    plt.close()

def plot_gait_diagram(contact_states: np.ndarray, sim_times: np.ndarray, reset_times: list[float], foot_labels: list[str], output_path: str, spacing: float = 1.0) -> plt.Figure:
    T, F = contact_states.shape
    assert sim_times.shape[0] == T, "sim_times length must match contact_states"

    fig, ax = plt.subplots(figsize=(180, F * 1.2))
    ax.set_xlabel('Time / s')
    ax.set_title('Gait Diagram with Air and Contact Times (white text = contact/stance phase, black text = air/swing phase)', fontsize=14)

    for reset_time in reset_times:
            ax.axvline(x=reset_time, linestyle=":", linewidth=1, color="orange", label='reset' if reset_time == reset_times[0] else None)

    for i, label in enumerate(foot_labels):
        y0 = i * spacing
        in_contact = contact_states[:, i].astype(bool)

        # Contact segments
        contact_segs = []
        start = None
        for t, c in enumerate(in_contact):
            if c and start is None:
                start = t
            elif not c and start is not None:
                contact_segs.append((start, t))
                start = None
        if start is not None:
            contact_segs.append((start, T))

        # Plot contact
        for s, e in contact_segs:
            ax.fill_between(sim_times[s:e], y0, y0 + spacing * 0.8, step='post', alpha=0.8, label=label + " contact phase" if s == contact_segs[0][0] else None)
            t_start = sim_times[s]
            t_end   = sim_times[e - 1]
            duration = t_end - t_start
            t_mid = 0.5 * (t_start + t_end)
            y_text = y0 + spacing * 0.3
            ax.text(t_mid, y_text, f"{duration:.3f}s", ha='center', va='center', color='white', fontsize=6, rotation=90)

        # Air segments
        air_segs = []
        if contact_segs and contact_segs[0][0] > 0:
            air_segs.append((0, contact_segs[0][0]))
        for (s0, e0), (s1, e1) in zip(contact_segs, contact_segs[1:]):
            air_segs.append((e0, s1))
        if contact_segs and contact_segs[-1][1] < T:
            air_segs.append((contact_segs[-1][1], T))

        # Annotate durations
        for a, b in air_segs:
            t_start = sim_times[a]
            t_end   = sim_times[b - 1]
            duration = t_end - t_start
            t_mid = 0.5 * (t_start + t_end)
            ax.text(t_mid, y0 + spacing * 0.5, f"{duration:.3f}s", ha='center', va='center', fontsize=6, rotation=90)

    ax.set_xticks(np.arange(0, sim_times[-1], 1))
    ax.set_yticks([i * spacing for i in range(F)])
    ax.set_yticklabels(foot_labels)
    ax.margins(x=0.005)
    ax.set_ylim(-spacing * 0.5, (F - 1) * spacing + spacing)
    ax.grid(axis='x', linestyle=':')
    ax.legend(loc='upper right', ncol=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600)
    return fig

def get_leg_linestyle(joint_name):
    if joint_name.startswith("FL"):
        return "solid"
    elif joint_name.startswith("FR"):
        return "dotted"
    elif joint_name.startswith("RL") or joint_name.startswith("HL"):
        return "dashed"
    elif joint_name.startswith("RR") or joint_name.startswith("HR"):
        return "dashdot"

# recognised prefixes for the four legs
_leg_prefixes = [
    ("FL_",),                     	# 0 = front-left
    ("FR_",),                     	# 1 = front-right
    ("RL_", "HL_"),               	# 2 = rear/​hind-left
    ("RR_", "HR_"),               	# 3 = rear/​hind-right
]

# -- accepted substrings for each joint "column" --------
JOINT_TYPE_SYNONYMS = {
    0: ("hip",  "haa"),      	# 0th column  = hip  / HAA  (ab-ad)
    1: ("thigh","hfe"),      	# 1st column  = thigh/ HFE  (flex-ext)
    2: ("calf", "kfe"),      	# 2nd column  = calf / KFE  (knee flex-ext)
}

def _column_from_name(jname: str) -> int | None:
    """
    Return 0,1,2 depending on which set of synonyms the name matches, else None.
    """
    low = jname.lower()
    for col, keys in JOINT_TYPE_SYNONYMS.items():
        if any(k in low for k in keys):
            return col
    raise ValueError("Could not determine joint row/col for plotting based on names")

# ----------------------------------------------------------------------------------------------------------------------
# Top-level plotting helpers (must be picklable for multiprocessing)
# ----------------------------------------------------------------------------------------------------------------------

def _plot_foot_contact_force_per_foot(sim_times, contact_forces_array, foot_labels, contact_state_array, reset_times, constraint_bounds, output_dir, pickle_dir, linewidth):
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        ax.plot(sim_times, contact_forces_array[:, i],
                label=f'force_mag_{foot_labels[i]}', linewidth=linewidth)
        draw_limits(ax, "foot_contact_force", constraint_bounds)
        draw_resets(ax, reset_times)

        in_contact = contact_state_array[:, i].astype(bool)
        segments, start_idx = [], None
        for idx, val in enumerate(in_contact):
            if val and start_idx is None:
                start_idx = idx
            elif not val and start_idx is not None:
                segments.append((start_idx, idx)); start_idx = None
        if start_idx is not None:
            segments.append((start_idx, len(sim_times)))

        first = True
        for s, e in segments:
            ax.axvspan(sim_times[s], sim_times[e-1], facecolor='gray', alpha=0.3,
                       label='in contact' if first else None)
            first = False

        ax.set_title(f"Foot contact force magnitude {foot_labels[i]}", fontsize=16)
        ax.set_ylabel('Force / N')
        ax.legend()
    fig.tight_layout()
    pdf = os.path.join(output_dir, "foot_contact_forces",'foot_contact_force_each.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, 'foot_contact_force_each.pickle'), 'wb') as f:
        pickle.dump(fig, f)

def _plot_hist_contact_forces_grid(contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE):
    """
    Plots a 2×2 grid of histograms of contact forces for each foot, showing only forces > 0 N.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("Histogram of Foot Contact Forces", fontsize=18)
    
    # Loop over each foot label / column
    for i, label in enumerate(foot_labels):
        ax = axes.flat[i]
        
        forces = contact_forces_array[:, i]
        positive_forces = forces[forces > 0]
        
        if positive_forces.size == 0:
            # Warn if no positive forces were found
            ax.text(0.5, 0.5, "No forces > 0 N", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="red")
        else:
            ax.hist(positive_forces, bins='auto', alpha=0.7)
        
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Force / N")
        ax.set_ylabel("Count")
    
    fig.tight_layout(rect=(0, 0, 1, 0.96)) # leave space for the suptitle
    
    pdf_path = os.path.join(output_dir, "foot_contact_forces", "hist_contact_forces_grid.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, dpi=600)
    
    pickle_path = os.path.join(pickle_dir, "hist_contact_forces_grid.pickle")
    with open(pickle_path, 'wb') as f:
        pickle.dump(fig, f)

def _plot_hist_contact_forces_overview(contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE):
    """
    Plots an overview histogram of contact forces for each foot on the same axes, showing only forces > 0 N.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    for i, label in enumerate(foot_labels):
        forces = contact_forces_array[:, i]
        positive_forces = forces[forces > 0]
        
        if positive_forces.size > 0:
            ax.hist(positive_forces, bins='auto', alpha=0.6, label=label)
        else:
            # Warn in the plot that there's no data for this label
            ax.text(0.5, 0.5 - 0.05 * i, f"No >0 data for '{label}'", transform=ax.transAxes, fontsize=8, color='gray', ha='center')
    
    ax.set_title("Histogram of Foot Contact Forces", fontsize=16)
    ax.set_xlabel("Force / N")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right', fontsize=8)
    
    fig.tight_layout()
    
    pdf_path = os.path.join(output_dir, "foot_contact_forces", "hist_contact_forces_overview.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, dpi=600)
    
    pickle_path = os.path.join(pickle_dir, "hist_contact_forces_overview.pickle")
    with open(pickle_path, 'wb') as f:
        pickle.dump(fig, f)

def _plot_joint_metric(metric_name, data_arr, sim_times, joint_names, leg_row, leg_col, foot_from_joint, contact_state_array, reset_times, constraint_bounds, metric_to_constraint_term_mapping, metric_to_unit_mapping, output_dir, pickle_dir, linewidth):
    fig, axes = plt.subplots(4, 3, sharex=True, figsize=(18, 12))
    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        if row is None or col is None:
            raise ValueError("Could not determine joint row/col for plotting")
        ax = axes[row, col]
        ax.plot(sim_times, data_arr[:, j], linewidth=linewidth)
        if metric_name == 'position':
            draw_limits(ax, jn, constraint_bounds)
        else:
            draw_limits(ax, metric_to_constraint_term_mapping[metric_name], constraint_bounds)
        draw_resets(ax, reset_times)

        fid = foot_from_joint[j]
        if fid is not None:
            in_c = contact_state_array[:, fid].astype(bool)
            start = None
            for t, val in enumerate(in_c):
                if val and start is None: start = t
                if (not val or t == len(in_c)-1) and start is not None:
                    end = t if not val else t+1
                    ax.axvspan(sim_times[start], sim_times[end-1], facecolor='gray', alpha=0.5)
                    start = None
        ax.set_title(f"Joint {metric_name.replace('_', ' ')} for {jn}", fontsize=16)
        ax.set_ylabel(f"{metric_name.capitalize()} / {metric_to_unit_mapping[metric_name]}")

    axes[-1, 0].set_xlabel('Time / s')
    fig.tight_layout()
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f'joint_{metric_name}_grid.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, f'joint_{metric_name}_grid.pickle'), 'wb') as f:
        pickle.dump(fig, f)

    # overview
    fig_ov, ax = plt.subplots(figsize=(12, 6))
    for j in range(data_arr.shape[1]):
        ax.plot(sim_times, data_arr[:, j], label=joint_names[j], linewidth=linewidth, linestyle=get_leg_linestyle(joint_names[j]))
    if metric_name == 'position':
        for jn in joint_names:
            draw_limits(ax, jn, constraint_bounds)
    else:
        draw_limits(ax, metric_to_constraint_term_mapping[metric_name], constraint_bounds)
    draw_resets(ax, reset_times)
    ax.set_xlabel('Time / s')
    ax.set_title(f"Joint {metric_name.replace('_', ' ')} overview", fontsize=16)
    ax.set_ylabel(f"{metric_name.capitalize()} / {metric_to_unit_mapping[metric_name]}")
    ax.legend(loc='upper right', ncol=2)
    fig_ov.tight_layout()
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f'joint_{metric_name}_overview.pdf')
    fig_ov.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, f'joint_{metric_name}_overview.pickle'), 'wb') as f:
        pickle.dump(fig_ov, f)

def _plot_hist_joint_grid(metric_name, data_arr, joint_names, leg_row, leg_col, metric_to_unit_mapping, output_dir, pickle_dir):
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle(f"Histogram of Joint {metric_name.replace('_', ' ').title()}", fontsize=18)
    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        ax = axes[row, col]
        ax.hist(data_arr[:, j], bins='auto', alpha=0.7)
        ax.set_title(jn, fontsize=12)
        ax.set_xlabel(f"{metric_name.capitalize()} / {metric_to_unit_mapping[metric_name]}")
        ax.set_ylabel("Count")
    fig.tight_layout()
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"hist_joint_{metric_name}_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, f"hist_joint_{metric_name}_grid.pickle"), 'wb') as f:
        pickle.dump(fig, f)

def _plot_hist_joint_metric(metric_name, data_arr, joint_names, metric_to_unit_mapping, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for j, jn in enumerate(joint_names):
        ax.hist(data_arr[:, j], bins='auto', alpha=0.6, label=jn)
    ax.set_title(f"Histogram of joint {metric_name.replace('_',' ')}", fontsize=16)
    ax.set_xlabel(f"{metric_name.capitalize()} / {metric_to_unit_mapping[metric_name]}")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"hist_joint_{metric_name}_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, f"hist_joint_{metric_name}_overview.pickle"), 'wb') as f:
        pickle.dump(fig, f)

def _plot_hist_air_time_per_foot_grid(air_segments_per_foot, foot_labels, output_dir, pickle_dir, FIGSIZE):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("Histogram of Air-Time Durations per Foot", fontsize=18)
    for i, label in enumerate(foot_labels):
        durations = air_segments_per_foot[label]
        ax = axes.flat[i]
        if durations:
            ax.hist(durations, bins='auto', alpha=0.7)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Air Time / s")
        ax.set_ylabel("Count")
    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", "air_time", "hist_air_time_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, "hist_air_time_grid.pickle"), 'wb') as f:
        pickle.dump(fig, f)

def _plot_hist_air_time_per_foot_single(label, durations, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if durations:
        ax.hist(durations, bins='auto', alpha=0.7)
    ax.set_title(f"Histogram of Air-Time Durations ({label})", fontsize=16)
    ax.set_xlabel("Air Time / s")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    safe_label = label.replace(' ', '_')
    pdf = os.path.join(output_dir, "aggregates", "air_time", f"hist_air_time_{safe_label}.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, f"hist_air_time_{safe_label}.pickle"), 'wb') as f:
        pickle.dump(fig, f)

def _plot_combined_energy(sim_times, combined_energy, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, combined_energy, label='total_energy', linewidth=linewidth)
    draw_resets(ax, reset_times)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Energy / J')
    ax.set_title('Total Cumulative Joint Energy', fontsize=16)
    ax.legend()
    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", 'combined_energy_overview.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, 'combined_energy_overview.pickle'), 'wb') as f:
        pickle.dump(fig, f)

def _plot_cost_of_transport(sim_times, cost_of_transport_time_series, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, cost_of_transport_time_series, label='cost_of_transport', linewidth=linewidth)
    draw_resets(ax, reset_times)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Cost of Transport / -')
    ax.set_title('Instantaneous Cost of Transport', fontsize=16)
    ax.set_ylim(0, 6)
    ax.legend()
    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", "cost_of_transport", 'cost_of_transport_over_time.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, 'cost_of_transport_over_time.pickle'), 'wb') as f:
        pickle.dump(fig, f)

def _plot_hist_cost_of_transport(cost_of_transport_time_series, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(cost_of_transport_time_series[~np.isnan(cost_of_transport_time_series)], bins='auto', alpha=0.7)
    ax.set_xlabel('Cost of Transport / -')
    ax.set_xlim(0, 6)
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Cost of Transport', fontsize=16)
    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", "cost_of_transport", 'hist_cot_overview.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, 'hist_cot_overview.pickle'), 'wb') as f:
        pickle.dump(fig, f)

def _plot_combined_base_position(sim_times, base_position_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig_bp, axes_bp = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, axis_label in enumerate(['X', 'Y', 'Z']):
        axes_bp[i].plot(sim_times, base_position_array[:, i], label=f'position_{axis_label}', linewidth=linewidth)
        draw_resets(axes_bp[i], reset_times)
        axes_bp[i].set_title(f'World Position {axis_label}')
        axes_bp[i].set_ylabel('Position / m')
        axes_bp[i].legend()
        axes_bp[i].grid(True)
    axes_bp[-1].set_xlabel('Time / s')
    fig_bp.tight_layout()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_position_subplots_world.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_bp.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, 'base_position_subplots_world.pickle'), 'wb') as f:
        pickle.dump(fig_bp, f)

def _plot_combined_orientation(sim_times, base_orientation_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig_bo, axes_bo = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, orient_label in enumerate(['Yaw', 'Pitch', 'Roll']):
        axes_bo[i].plot(sim_times, base_orientation_array[:, i], label=orient_label, linewidth=linewidth)
        draw_resets(axes_bo[i], reset_times)
        axes_bo[i].set_ylabel(f'{orient_label} / rad')
        axes_bo[i].set_title(f'World Orientation {orient_label}')
        axes_bo[i].legend()
        axes_bo[i].grid(True)
    axes_bo[-1].set_xlabel('Time / s')
    fig_bo.tight_layout()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_orientation_subplots.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_bo.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, 'base_orientation_subplots.pickle'), 'wb') as f:
        pickle.dump(fig_bo, f)

def _plot_combined_base_velocity(sim_times, base_linear_velocity_array, commanded_velocity_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig_blv, axes_blv = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, vel_label in enumerate(['Velocity X', 'Velocity Y', 'Velocity Z']):
        axes_blv[i].plot(sim_times, base_linear_velocity_array[:, i], label=vel_label, linewidth=linewidth)
        if i != 2:
            axes_blv[i].plot(sim_times, commanded_velocity_array[:, i], linestyle='--', label=f'cmd_{vel_label}', linewidth=linewidth)
        draw_resets(axes_blv[i], reset_times)
        axes_blv[i].set_ylabel(f'{vel_label} / m * sec^(-1)')
        axes_blv[i].set_title(f'Base Linear {vel_label}')
        axes_blv[i].legend()
        axes_blv[i].grid(True)
    axes_blv[-1].set_xlabel('Time / s')
    fig_blv.tight_layout()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_linear_velocity_subplots.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_blv.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, 'base_linear_velocity_subplots.pickle'), 'wb') as f:
        pickle.dump(fig_blv, f)

def _plot_combined_base_angular_velocities(sim_times, base_angular_velocity_array, commanded_velocity_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig_bav, axes_bav = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, vel_label in enumerate(['Omega X', 'Omega Y', 'Omega Z']):
        axes_bav[i].plot(sim_times, base_angular_velocity_array[:, i], label=vel_label, linewidth=linewidth)
        if i == 2:
            axes_bav[i].plot(sim_times, commanded_velocity_array[:, i], linestyle='--', label=f'cmd_{vel_label}', linewidth=linewidth)
        draw_resets(axes_bav[i], reset_times)
        axes_bav[i].set_ylabel(f'{vel_label} / rad * sec^(-1)')
        axes_bav[i].set_title(f'Base {vel_label}')
        axes_bav[i].legend()
        axes_bav[i].grid(True)
    axes_bav[-1].set_xlabel('Time / s')
    fig_bav.tight_layout()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_angular_velocity_subplots.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_bav.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, 'base_angular_velocity_subplots.pickle'), 'wb') as f:
        pickle.dump(fig_bav, f)

def _plot_total_base_overview(sim_times, base_position_array, base_orientation_array, base_linear_velocity_array, base_angular_velocity_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig_overview, overview_axes = plt.subplots(2, 2, figsize=(20, 16))
    arrays = [base_position_array, base_orientation_array, base_linear_velocity_array, base_angular_velocity_array]
    titles = ['Base World Position', 'Base Orientation', 'Base Linear Velocity', 'Base Angular Velocity']
    labels = [['X', 'Y', 'Z'], ['Yaw', 'Pitch', 'Roll'], ['VX', 'VY', 'VZ'], ['WX', 'WY', 'WZ']]
    for ax, arr, title, axis_labels in zip(overview_axes.flatten(), arrays, titles, labels):
        for i, lbl in enumerate(axis_labels):
            ax.plot(sim_times, arr[:, i], label=lbl, linewidth=linewidth)
            draw_resets(ax, reset_times)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Time / s')
        ax.legend()
        ax.grid(True)
    fig_overview.tight_layout()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_overview_world.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_overview.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, 'base_overview_world.pickle'), 'wb') as f:
        pickle.dump(fig_overview, f)

def _plot_gait_diagram(contact_state_array, sim_times, reset_times, foot_labels, output_dir, pickle_dir):
    fig = plot_gait_diagram(contact_state_array, sim_times, reset_times, foot_labels, os.path.join(output_dir, "aggregates", 'gait_diagram.pdf'), spacing=1.0)
    with open(os.path.join(pickle_dir, 'gait_diagram.pickle'), 'wb') as f:
        pickle.dump(fig, f)

# ----------------------------------------------------------------------------------------------------------------------
# box-plot helpers
# ----------------------------------------------------------------------------------------------------------------------

def _plot_box_joint_grid(metric_name, data_arr, joint_names, leg_row, leg_col, metric_to_unit_mapping, output_dir, pickle_dir):
    """
    4×3 grid of box plots, one per joint, for the given metric.
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle(f"box Plot of Joint {metric_name.replace('_', ' ').title()}", fontsize=18)

    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        ax = axes[row, col]

        ax.boxplot(x=data_arr[:, j], showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(jn, fontsize=12)
        ax.set_xlabel(f"{metric_name.capitalize()} / {metric_to_unit_mapping[metric_name]}", fontsize=8)

    fig.tight_layout()
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"box_joint_{metric_name}_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, f"box_joint_{metric_name}_grid.pickle"), "wb") as f:
        pickle.dump(fig, f)


def _plot_box_joint_metric(metric_name, data_arr, joint_names, metric_to_unit_mapping, output_dir, pickle_dir, FIGSIZE):
    """
    Overview box plot (all joints on one axis) for the given metric.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.boxplot(x=[data_arr[:, j] for j in range(data_arr.shape[1])], positions=np.arange(1, len(joint_names) + 1), showmeans=True, showcaps=True, showbox=True, showfliers=False)

    ax.set_title(f"Box Plot of Joint {metric_name.replace('_', ' ')}", fontsize=16)
    ax.set_xlabel("Joint")
    ax.set_ylabel(f"{metric_name.capitalize()} / {metric_to_unit_mapping[metric_name]}")
    ax.set_xticks(np.arange(1, len(joint_names) + 1))
    ax.set_xticklabels(joint_names, rotation=45, ha="right")
    fig.tight_layout()

    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"box_joint_{metric_name}_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, f"box_joint_{metric_name}_overview.pickle"), "wb") as f:
        pickle.dump(fig, f)


def _plot_box_contact_forces_grid(contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE):
    """
    2×2 grid of box plots of contact-force magnitudes (>0 N) per foot.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("box Plot of Foot Contact Forces", fontsize=18)

    for i, label in enumerate(foot_labels):
        ax = axes.flat[i]
        forces = contact_forces_array[:, i]
        positive = forces[forces > 0]
        if positive.size == 0:
            ax.text(0.5, 0.5, "No forces > 0 N", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="red")
        else:
            ax.boxplot(positive, showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Force / N")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf = os.path.join(output_dir, "foot_contact_forces", "box_contact_forces_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, "box_contact_forces_grid.pickle"), "wb") as f:
        pickle.dump(fig, f)


def _plot_box_contact_forces_overview(contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE):
    """
    Overview box plot of contact-force magnitudes (>0 N) for all feet.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    data, used_labels = [], []
    for i, label in enumerate(foot_labels):
        positive = contact_forces_array[:, i][contact_forces_array[:, i] > 0]
        if positive.size > 0:
            data.append(positive)
            used_labels.append(label)

    if data:
        ax.boxplot(data, positions=np.arange(1, len(data) + 1), showmeans=True, showcaps=True, showbox=True, showfliers=False)
    ax.set_title("Box Plot of Foot Contact Forces", fontsize=16)
    ax.set_xlabel("Foot")
    ax.set_ylabel("Force / N")
    ax.set_xticks(np.arange(1, len(used_labels) + 1))
    ax.set_xticklabels(used_labels)
    fig.tight_layout()

    pdf = os.path.join(output_dir, "foot_contact_forces", "box_contact_forces_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, "box_contact_forces_overview.pickle"), "wb") as f:
        pickle.dump(fig, f)


def _plot_box_air_time_per_foot_grid(air_segments_per_foot, foot_labels, output_dir, pickle_dir, FIGSIZE):
    """
    2×2 grid of box plots of air-time durations per foot.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("box Plot of Air-Time Durations per Foot", fontsize=18)

    for i, label in enumerate(foot_labels):
        durations = air_segments_per_foot[label]
        ax = axes.flat[i]
        if durations:
            ax.boxplot(durations, showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Air Time / s")

    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", "air_time", "box_air_time_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, "box_air_time_grid.pickle"), "wb") as f:
        pickle.dump(fig, f)


def _plot_box_air_time_per_foot_single(label, durations, output_dir, pickle_dir, FIGSIZE):
    """
    Single-foot box plot of air-time durations.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if durations:
        ax.boxplot(durations, showmeans=True, showcaps=True, showbox=True, showfliers=False)
    else:
        ax.text(0.5, 0.5, "No air-time segments", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="red")

    ax.set_title(f"Box Plot of Air-Time Durations ({label})", fontsize=16)
    ax.set_xlabel("Air Time / s")
    fig.tight_layout()

    safe_label = label.replace(" ", "_")
    pdf = os.path.join(output_dir, "aggregates", "air_time", f"box_air_time_{safe_label}.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, f"box_air_time_{safe_label}.pickle"), "wb") as f:
        pickle.dump(fig, f)


def _plot_box_cost_of_transport(cost_of_transport_time_series, output_dir, pickle_dir, FIGSIZE):
    """
    box plot of instantaneous cost-of-transport values.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    data = cost_of_transport_time_series[~np.isnan(cost_of_transport_time_series)]
    if data.size > 0:
        ax.boxplot(data, showmeans=True, showcaps=True, showbox=True, showfliers=False)
    ax.set_xlabel("Cost of Transport")
    ax.set_title("Box Plot of Cost of Transport", fontsize=16)
    fig.tight_layout()

    pdf = os.path.join(output_dir, "aggregates", "cost_of_transport", "box_cot_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    with open(os.path.join(pickle_dir, "box_cot_overview.pickle"), "wb") as f:
        pickle.dump(fig, f)

# ----------------------------------------------------------------------------------------------------------------------
# Main plot generation orchestrator
# ----------------------------------------------------------------------------------------------------------------------

def generate_plots(data, metrics, output_dir, interactive=False):
    """
    Recreate all plots from loaded data.
    - data: numpy.lib.npyio.NpzFile containing arrays
    - metrics: dict loaded from metrics_summary.json
    """
    os.makedirs(output_dir, exist_ok=True)
    pickle_dir = os.path.join(output_dir, "plot_figures_serialized")
    os.makedirs(pickle_dir, exist_ok=True)

    start = time.time()

    linewidth = 1
    FIGSIZE = (16, 9)

    # Unpack arrays
    sim_times = data['sim_times']
    reset_times = data['reset_times'].tolist()
    contact_forces_array = data['contact_forces_array']
    foot_labels = list(data['foot_labels'])
    contact_state_array = data['contact_state_array']
    constraint_bounds = data['constraint_bounds'].item()
    joint_names = list(data['joint_names'])
    air_segments_per_foot = data['air_segments_per_foot'].item()
    combined_energy = data['combined_energy']
    cost_of_transport_time_series = data['cost_of_transport_time_series']

    # build two look-up tables
    leg_row  	= [None] * len(joint_names) # index 0-3
    leg_col  	= [None] * len(joint_names) # index 0-2
    foot_from_joint = [None] * len(joint_names)

    for j, name in enumerate(joint_names):
        # find which leg
        for row, prefixes in enumerate(_leg_prefixes):
            if any(name.startswith(p) for p in prefixes):
                leg_row[j] = row
                foot_from_joint[j] = row # same index as contact_state columns
                break
        # find column inside that leg
        leg_col[j] = _column_from_name(name)

    print("joint names: ", joint_names)
    print("leg_row:", leg_row)
    print("leg_col:", leg_col)
    print("foot_from_joint:", foot_from_joint)

    # Build metrics dict
    metrics = {
        'position': data['joint_positions_array'],
        'velocity': data['joint_velocities_array'],
        'acceleration': data['joint_accelerations_array'],
        'torque': data['joint_torques_array'],
        'action_rate': data['action_rate_array'],
        'energy': data['energy_per_joint'],
        'power': data['power_array'],
    }

    metric_to_constraint_term_mapping = {
        'position': None,
        'velocity':	'joint_velocity',
        'acceleration': 'joint_acceleration',
        'torque': 'joint_torque',
        'action_rate': 'action_rate',
        'energy': None,
        'power': None
    }

    metric_to_unit_mapping = {
        'position': 'rad',
        'velocity':	'rad*s^(-1)',
        'acceleration': 'rad * s^(-2) ',
        'torque': 'Nm',
        'action_rate': 'rad * s^(-1)',
        'energy': 'J',
        'combined_energy': 'J',
        'cost_of_transport': '-',
        'power': 'W'
    }

    # Helper for mapping joints to feet
    foot_from_joint = []
    for name in joint_names:
        if   name.startswith('FL_'): foot_from_joint.append(0)
        elif name.startswith('FR_'): foot_from_joint.append(1)
        elif name.startswith('RL_') or name.startswith('HL_'): foot_from_joint.append(2)
        elif name.startswith('RR_') or name.startswith('HR_'): foot_from_joint.append(3)
        else: foot_from_joint.append(None)

    futures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 1) Foot contact-force per-foot grid
        futures.append(
            executor.submit(
                _plot_foot_contact_force_per_foot,
                sim_times, contact_forces_array, foot_labels,
                contact_state_array, reset_times, constraint_bounds,
                output_dir, pickle_dir, linewidth
            )
        )

        # 2) Joint metrics (grid + overview)
        for metric_name, data_arr in metrics.items():
            futures.append(
                executor.submit(
                    _plot_joint_metric,
                    metric_name, data_arr, sim_times, joint_names, leg_row, leg_col,
                    foot_from_joint, contact_state_array, reset_times, constraint_bounds,
                    metric_to_constraint_term_mapping, metric_to_unit_mapping,
                    output_dir, pickle_dir, linewidth
                )
            )

        # Histograms 4x3 per joint metric
        for mname, data_arr in metrics.items():
            futures.append(
                executor.submit(
                    _plot_hist_joint_grid,
                    mname, data_arr, joint_names, leg_row, leg_col,
                    metric_to_unit_mapping, output_dir, pickle_dir
                )
            )

        # 2x2 Hist contact forces
        futures.append(
            executor.submit(
                _plot_hist_contact_forces_grid,
                contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE
            )
        )

        # 2x2 Hist air-time per foot
        futures.append(
            executor.submit(
                _plot_hist_air_time_per_foot_grid,
                air_segments_per_foot, foot_labels, output_dir, pickle_dir, FIGSIZE
            )
        )

        # Hist overview per joint metric
        for metric_name, data_arr in metrics.items():
            futures.append(
                executor.submit(
                    _plot_hist_joint_metric,
                    metric_name, data_arr, joint_names,
                    metric_to_unit_mapping, output_dir, pickle_dir, FIGSIZE
                )
            )

        # Hist contact forces overview
        futures.append(
            executor.submit(
                _plot_hist_contact_forces_overview,
                contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE
            )
        )

        # Combined energy over time
        futures.append(
            executor.submit(
                _plot_combined_energy,
                sim_times, combined_energy, reset_times,
                output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )

        # Instantaneous cost of transport over time
        futures.append(
            executor.submit(
                _plot_cost_of_transport,
                sim_times, cost_of_transport_time_series, reset_times,
                output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )

        # Histogram of cost of transport
        futures.append(
            executor.submit(
                _plot_hist_cost_of_transport,
                cost_of_transport_time_series, output_dir, pickle_dir, FIGSIZE
            )
        )

        # Hist air-time per foot separate
        for label, durations in air_segments_per_foot.items():
            futures.append(
                executor.submit(
                    _plot_hist_air_time_per_foot_single,
                    label, durations, output_dir, pickle_dir, FIGSIZE
                )
            )

        # Base plots
        futures.append(
            executor.submit(
                _plot_combined_base_position,
                sim_times, data['base_position_array'], reset_times,
                output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )
        futures.append(
            executor.submit(
                _plot_combined_orientation,
                sim_times, data['base_orientation_array'], reset_times,
                output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )
        futures.append(
            executor.submit(
                _plot_combined_base_velocity,
                sim_times, data['base_linear_velocity_array'], data['commanded_velocity_array'],
                reset_times, output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )
        futures.append(
            executor.submit(
                _plot_combined_base_angular_velocities,
                sim_times, data['base_angular_velocity_array'], data['commanded_velocity_array'],
                reset_times, output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )
        futures.append(
            executor.submit(
                _plot_total_base_overview,
                sim_times, data['base_position_array'], data['base_orientation_array'],
                data['base_linear_velocity_array'], data['base_angular_velocity_array'],
                reset_times, output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )

        # Gait diagram
        futures.append(
            executor.submit(
                _plot_gait_diagram,
                contact_state_array, sim_times, reset_times,
                foot_labels, output_dir, pickle_dir
            )
        )

        # a) Joint-level box plots (grid + overview)
        for metric_name, data_arr in metrics.items():
            futures.append(
                executor.submit(
                    _plot_box_joint_grid,
                    metric_name, data_arr, joint_names, leg_row, leg_col,
                    metric_to_unit_mapping, output_dir, pickle_dir
                )
            )
            futures.append(
                executor.submit(
                    _plot_box_joint_metric,
                    metric_name, data_arr, joint_names,
                    metric_to_unit_mapping, output_dir, pickle_dir, FIGSIZE
                )
            )

        # b) Contact-force box plots
        futures.append(
            executor.submit(
                _plot_box_contact_forces_grid,
                contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_box_contact_forces_overview,
                contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE
            )
        )

        # c) Air-time box plots (grid + per-foot)
        futures.append(
            executor.submit(
                _plot_box_air_time_per_foot_grid,
                air_segments_per_foot, foot_labels, output_dir, pickle_dir, FIGSIZE
            )
        )
        for label, durations in air_segments_per_foot.items():
            futures.append(
                executor.submit(
                    _plot_box_air_time_per_foot_single,
                    label, durations, output_dir, pickle_dir, FIGSIZE
                )
            )

        # d) Cost-of-transport box plot
        futures.append(
            executor.submit(
                _plot_box_cost_of_transport,
                cost_of_transport_time_series, output_dir, pickle_dir, FIGSIZE
            )
        )

        # Ensure all tasks complete
        for f in futures:
            f.result()

    end = time.time()
    print(f"Plot generation took {(end-start):.4f} seconds.")

    # Interactive display if requested
    if interactive:
        plt.ion()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from saved simulation data.")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the .npz file containing recorded sim data.")
    parser.add_argument("--summary", type=str, default=None,
                        help="Path to metrics_summary.json (optional).")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Show figures interactively in addition to saving.")
    args = parser.parse_args()

    data, metrics = load_data(args.data_file, args.summary)
    generate_plots(data, metrics, os.path.dirname(args.data_file), interactive=args.interactive)

if __name__ == "__main__":
    main()
