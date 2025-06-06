"""
Takes numpy buffers stored on disk and generates interactive plots, saving them to disk using picke.dump to be able to open them after generation.
"""

import argparse
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import matplotlib.animation as animation
import time

from metrics_utils import (
    compute_energy_arrays,
    compute_swing_durations,
    compute_swing_heights,
    compute_swing_lengths,
    summarize_metric,
    optimal_bin_edges,
    compute_histogram,
    compute_stance_segments,
    compute_swing_segments,
    compute_trimmed_histogram_data
)

# Disable interactive display until --interactive is set
plt.ioff()

def load_data(npz_path, summary_path=None):
    """
    Load simulation data from a .npz file.
    """
    data = np.load(npz_path, allow_pickle=True)
    return data

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

def _array_to_metric_dict(arr: np.ndarray, foot_labels: list[str]) -> dict[str, list[float]]:
    """
    Converts (T, 4) array to {'FL': [...], 'FR': [...], ...}
    suitable for _plot_hist_metric_* and _plot_box_metric_* helpers.
    """
    return {lbl: arr[:, i].tolist() for i, lbl in enumerate(foot_labels)}

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
    ax.set_xlabel('Time (s)')
    ax.set_title('Gait Diagram with Air and Contact Times (white text = contact/stance phase, black text = air/swing phase)', fontsize=14)

    for reset_time in reset_times:
            ax.axvline(x=reset_time, linestyle=":", linewidth=1, color="orange", label='reset' if reset_time == reset_times[0] else None)

    for i, label in enumerate(foot_labels):
        y0 = i * spacing
        in_contact = contact_states[:, i].astype(bool)
        contact_segments = compute_stance_segments(in_contact)

        # Plot contact
        for s, e in contact_segments:
            ax.fill_between(sim_times[s:e], y0, y0 + spacing * 0.8, step='post', alpha=0.8, label=label + " stance" if s == contact_segments[0][0] else None)
            t_start = sim_times[s]
            t_end   = sim_times[e - 1]
            duration = t_end - t_start
            t_mid = 0.5 * (t_start + t_end)
            y_text = y0 + spacing * 0.3
            ax.text(t_mid, y_text, f"{duration:.3f}s", ha='center', va='center', color='white', fontsize=6, rotation=90)

        swing_segments = compute_swing_segments(in_contact)

        # Annotate durations
        for a, b in swing_segments:
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

def _plot_body_frame_foot_position_heatmap(foot_positions_body_frame: np.ndarray, output_dir: str, pickle_dir: str, bin_count: int = 100, FIGSIZE: tuple[int, int] = (20, 20)):
    """
    Discretised heat-map of all foot XY positions in the *body* frame.

    Parameters
    ----------
    foot_positions_body_frame : np.ndarray
        (T, 4, 3) array - XY columns are used.
    output_dir : str
        Root output directory (same as the other helpers).
    pickle_dir : str
        Directory where the pickled Figure is stored.
    bin_count : int, optional
        Number of bins per axis (uniform). 100x100 gives ~1 cm² cells for a typical quadruped workspace (~±0.5 m).
    FIGSIZE : tuple[int, int], optional
        Figure size in inches.
    """
    xy = foot_positions_body_frame[:, :, :2].reshape(-1, 2) # (N, 2)
    x, y = xy[:, 0], xy[:, 1]

    x_edges = np.linspace(x.min(), x.max(), bin_count + 1)
    y_edges = np.linspace(y.min(), y.max(), bin_count + 1)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    # Use log-10 values if data range is high
    if counts.max() > 0 and counts.max() / counts[counts > 0].min() > 50:
        counts_display = np.log10(counts + 1)
        cbar_label = "log₁₀(occupancy + 1)"
    else:
        counts_display = counts
        cbar_label = "occupancy (samples)"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(counts_display.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="equal", cmap="viridis")
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("Body-X (m)")
    ax.set_ylabel("Body-Y (m)")
    ax.set_title("Foot-position occupancy heat-map (body frame, CoM, top-down)")

    pdf_dir = os.path.join(output_dir, "foot_positions_body_frame_com", "heatmap")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "foot_position_heatmap_body_frame_com.pdf")
    fig.savefig(pdf_path, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "foot_position_heatmap_body_frame_com.pickle"), "wb") as f:
            pickle.dump(fig, f)
    plt.close(fig)

def _plot_body_frame_foot_position_heatmap_grid(foot_positions_body_frame: np.ndarray, foot_labels: list[str], output_dir: str, pickle_dir: str, bin_count: int = 100, FIGSIZE: tuple[int, int] = (20, 20)):
    """
    2x2 grid of occupancy heat-maps - one per foot - in body frame.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("Foot-position heat-maps (body frame, per foot)", fontsize=18)

    for i, (ax, lbl) in enumerate(zip(axes.flat, foot_labels)):
        xy = foot_positions_body_frame[:, i, :2] # (T, 2)
        x, y = xy[:, 0], xy[:, 1]

        x_edges = np.linspace(x.min(), x.max(), bin_count + 1)
        y_edges = np.linspace(y.min(), y.max(), bin_count + 1)
        counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

        if counts.max() > 0 and counts.max() / counts[counts > 0].min() > 50:
            counts_display = np.log10(counts + 1)
            cbar_label = "log₁₀(+1)"
        else:
            counts_display = counts
            cbar_label = "occupancy"

        im = ax.imshow(counts_display.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="equal", cmap="viridis")
        ax.set_title(lbl)
        ax.set_xlabel("Body-X (m)")
        ax.set_ylabel("Body-Y (m)")
        fig.colorbar(im, ax=ax, label=cbar_label)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    pdf_dir = os.path.join(output_dir, "foot_positions_body_frame_com", "heatmap")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "foot_position_heatmap_body_frame_com_grid.pdf")
    fig.savefig(pdf_path, dpi=600)

    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "foot_position_heatmap_body_frame_com_grid.pickle"), "wb") as f:
            pickle.dump(fig, f)

    plt.close(fig)

def _plot_body_frame_foot_position_heatmap_single(foot_positions_body_frame: np.ndarray, foot_idx: int, foot_label: str, output_dir: str, pickle_dir: str, bin_count: int = 100, FIGSIZE: tuple[int, int] = (20, 20)):
    """
    Single-foot heat-map stored as
    .../foot_positions_body_frame/heatmap/<label>/foot_position_heatmap_<label>.pdf
    """
    xy = foot_positions_body_frame[:, foot_idx, :2]
    x, y = xy[:, 0], xy[:, 1]

    x_edges = np.linspace(x.min(), x.max(), bin_count + 1)
    y_edges = np.linspace(y.min(), y.max(), bin_count + 1)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    if counts.max() > 0 and counts.max() / counts[counts > 0].min() > 50:
        counts_display = np.log10(counts + 1)
        cbar_label = "log₁₀(+1)"
    else:
        counts_display = counts
        cbar_label = "occupancy"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(counts_display.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="equal", cmap="viridis")
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("Body-X (m)")
    ax.set_ylabel("Body-Y (m)")
    ax.set_title(f"Foot-position heat-map (body frame) – {foot_label}")

    safe_lbl = foot_label.replace(" ", "_")
    pdf_dir = os.path.join(output_dir, "foot_positions_body_frame_com", "heatmap", safe_lbl.lower())
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"foot_position_heatmap_{safe_lbl.lower()}_com.pdf")
    fig.savefig(pdf_path, dpi=600)
    
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"foot_position_heatmap_{safe_lbl.lower()}_com.pickle"), "wb") as f:
            pickle.dump(fig, f)

    plt.close(fig)

def _animate_body_frame_foot_positions(foot_positions_body_frame: np.ndarray, contact_state_array: np.ndarray, foot_labels: list[str], output_path: str, fps: int = 30):
    """
    Top-down animation of body-frame foot XY positions.
    - filled marker : stance / contact
    - hollow marker : swing / air
    """
    T = foot_positions_body_frame.shape[0]
    colours = ['red', 'blue', 'green', 'purple']

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel('Body-X (m)')
    ax.set_ylabel('Body-Y (m)')
    ax.set_title('Foot trajectories (body frame, CoM, top-down)')

    # create one PathCollection per foot
    scatters = [ax.scatter([], [], s=60, c=c, edgecolor=c, label=lbl) for c, lbl in zip(colours, foot_labels)]
    ax.legend(loc='upper right')

    # fixed limits (slightly padded) so blit tests don’t crop everything away
    x_min, x_max = np.min(foot_positions_body_frame[:, :, 0]), np.max(foot_positions_body_frame[:, :, 0])
    y_min, y_max = np.min(foot_positions_body_frame[:, :, 1]), np.max(foot_positions_body_frame[:, :, 1])
    padding = 0.05
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    def _init():
        # initialise each scatter with the first frame so it is non-empty
        for i, sc in enumerate(scatters):
            xy0 = foot_positions_body_frame[0, i, :2].reshape(1, 2)
            sc.set_offsets(xy0)
        return scatters

    def _animate(k: int):
        for i, sc in enumerate(scatters):
            xy = foot_positions_body_frame[k, i, :2].reshape(1, 2)
            sc.set_offsets(xy)
            if contact_state_array[k, i]:
                sc.set_facecolor(colours[i]) # filled (stance)
            else:
                sc.set_facecolor('none') # hollow (swing)
            sc.set_edgecolor(colours[i]) # always draw edges
        return scatters

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani = animation.FuncAnimation(fig, _animate, init_func=_init, frames=T, interval=1000 / fps, blit=True)
    ani.save(output_path, fps=fps)
    plt.close(fig)

def _plot_foot_position_time_series(
        positions_array: np.ndarray,
        axis_label: str,
        frame_label: str,
        sim_times: np.ndarray,
        foot_labels: list[str],
        contact_state_array: np.ndarray,
        reset_times: list[float],
        output_dir: str,
        pickle_dir: str,
        FIGSIZE: tuple[int, int],
        linewidth: float,
        subfolder: str
):
    # ---------- per-foot grid ----------
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=FIGSIZE)
    for i, ax in enumerate(axes.flat):
        ax.plot(sim_times, positions_array[:, i], linewidth=linewidth, label=f'{axis_label.lower()}_{foot_labels[i]}')
        ax.grid()
        first = True
        for start_timestep, end_timestep in compute_stance_segments(contact_state_array[:, i].astype(bool)):
            ax.axvspan(sim_times[start_timestep], sim_times[end_timestep - 1], color='gray', alpha=.3, label='in contact' if first else None)
            first = False
        draw_resets(ax, reset_times)
        ax.set_title(f'Foot {axis_label} ({frame_label.replace("_", " ")}) {foot_labels[i]} {"(CoM)" if frame_label == "body_frame" else "(toe tip)"}', fontsize=14)
        ax.set_ylabel(f'{axis_label} (m)')
        ax.legend()
    axes[-1, 0].set_xlabel('Time (s)')
    fig.tight_layout()

    subdir = os.path.join(output_dir, subfolder)
    os.makedirs(subdir, exist_ok=True)
    pdf = os.path.join(subdir, f'foot_pos_{axis_label.lower()}_each.pdf')
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'foot_pos_{axis_label.lower()}_each_{frame_label}.pickle'), 'wb') as f:
            pickle.dump(fig, f)

    # ---------- overview ----------
    fig_ov, ax = plt.subplots(figsize=(FIGSIZE[0], FIGSIZE[1]))
    for i, lbl in enumerate(foot_labels):
        ax.grid()
        ax.plot(sim_times, positions_array[:, i], label=lbl, linewidth=linewidth)
    draw_resets(ax, reset_times)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{axis_label} (m)')
    ax.set_title(f'Foot {axis_label} ({frame_label.replace("_", " ")}) {foot_labels[i]} {"(CoM)" if frame_label == "body_frame" else "(toe tip)"} overview', fontsize=14)
    ax.legend(ncol=2, loc='upper right')
    fig_ov.tight_layout()
    pdf = os.path.join(subdir, f'foot_pos_{axis_label.lower()}_overview.pdf')
    fig_ov.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'foot_pos_{axis_label.lower()}_overview_{frame_label}.pickle'), 'wb') as f:
            pickle.dump(fig_ov, f)

def _plot_hist_metric_grid(metric_dict: dict[str, list[float]], title: str, xlabel: str, foot_labels: list[str], output_dir: str, pickle_dir: str, subfolder: str, FIGSIZE: tuple[int, int]):
    """
    Generic 2x2 grid histogram for per-foot metrics in `metric_dict`.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=18)
    for i, lbl in enumerate(foot_labels):
        ax = axes.flat[i]
        ax.grid()
        data = metric_dict[lbl]
        if data:
            counts, edges = compute_trimmed_histogram_data(np.array(data))
            # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
            ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
        else:
            ax.text(.5, .5, 'no data', ha='center', va='center', transform=ax.transAxes, color='red')
        ax.set_title(lbl, fontsize=14)
        ax.set_xlabel(xlabel); ax.set_ylabel('Count')
    fig.tight_layout(rect=(0, 0, 1, .96))
    pdf = os.path.join(output_dir, subfolder, f"hist_{os.path.basename(subfolder)}_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"hist_{os.path.basename(subfolder)}_grid.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_hist_metric_overview(metric_dict: dict[str, list[float]], title: str, xlabel: str, foot_labels: list[str], output_dir: str, pickle_dir: str, subfolder: str, FIGSIZE: tuple[int, int]):
    """
    Generic overview histogram (all feet on same axes).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for lbl in foot_labels:
        data = metric_dict[lbl]
        if data:
            ax.grid()
            counts, edges = compute_trimmed_histogram_data(np.array(data))
            # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
            ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0, label=lbl)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right')
    fig.tight_layout()
    pdf = os.path.join(output_dir, subfolder, f"hist_{os.path.basename(subfolder)}_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"hist_{os.path.basename(subfolder)}_overview.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_box_metric_grid(metric_dict: dict[str, list[float]], title: str, xlabel: str, foot_labels: list[str], output_dir: str, pickle_dir: str, subfolder: str, FIGSIZE: tuple[int, int]):
    """
    Generic 2x2 grid box-plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=18)
    for i, lbl in enumerate(foot_labels):
        ax = axes.flat[i]
        ax.grid()
        data = metric_dict[lbl]
        if data:
            ax.boxplot(data, showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(lbl, fontsize=14); ax.set_xlabel(xlabel)
    fig.tight_layout()
    pdf = os.path.join(output_dir, subfolder, f"box_{os.path.basename(subfolder)}_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"box_{os.path.basename(subfolder)}_grid.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_box_metric_overview(metric_dict: dict[str, list[float]], title: str, xlabel: str, foot_labels: list[str], output_dir: str, pickle_dir: str, subfolder: str, FIGSIZE: tuple[int, int]):
    """
    Generic overview box-plot.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    data, lbls = [], []
    for lbl in foot_labels:
        if metric_dict[lbl]:
            data.append(metric_dict[lbl]); lbls.append(lbl)
    if data:
        ax.grid()
        ax.boxplot(data, positions=np.arange(1, len(lbls)+1), showmeans=True, showcaps=True, showbox=True, showfliers=False)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Foot'); ax.set_ylabel(xlabel)
    ax.set_xticks(np.arange(1, len(lbls)+1)); ax.set_xticklabels(lbls)
    fig.tight_layout()
    pdf = os.path.join(output_dir, subfolder, f"box_{os.path.basename(subfolder)}_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"box_{os.path.basename(subfolder)}_overview.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_foot_contact_force_per_foot(sim_times, contact_forces_array, foot_labels, contact_state_array, reset_times, constraint_bounds, output_dir, pickle_dir, linewidth):
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        ax.plot(sim_times, contact_forces_array[:, i], label=f'force_mag_{foot_labels[i]}', linewidth=linewidth)
        ax.grid()
        draw_limits(ax, "foot_contact_force", constraint_bounds)
        draw_resets(ax, reset_times)

        first = True
        for s, e in compute_stance_segments(in_contact=contact_state_array[:, i].astype(bool)):
            ax.axvspan(sim_times[s], sim_times[e-1], facecolor='gray', alpha=0.3, label='in contact' if first else None)
            first = False

        ax.set_title(f"Foot contact force magnitude {foot_labels[i]}", fontsize=16)
        ax.set_ylabel('Force (N)')
        ax.legend()
    fig.tight_layout()
    pdf = os.path.join(output_dir, "foot_contact_forces",'foot_contact_force_each.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
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
        ax.grid()
        forces = contact_forces_array[:, i]
        positive_forces = forces[forces > 0]
        
        if positive_forces.size == 0:
            # Warn if no positive forces were found
            ax.text(0.5, 0.5, "No forces > 0 N", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="red")
        else:
            counts, edges = compute_trimmed_histogram_data(positive_forces)
            # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
            ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
        
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Force (N)")
        ax.set_ylabel("Count")
    
    fig.tight_layout(rect=(0, 0, 1, 0.96)) # leave space for the suptitle
    
    pdf_path = os.path.join(output_dir, "foot_contact_forces", "hist_contact_forces_grid.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, dpi=600)
    
    pickle_path = os.path.join(pickle_dir, "hist_contact_forces_grid.pickle")
    if pickle_dir != "":
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
            ax.grid()
            counts, edges = compute_trimmed_histogram_data(positive_forces)
            # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
            ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0, label=label)
        else:
            # Warn in the plot that there's no data for this label
            ax.text(0.5, 0.5 - 0.05 * i, f"No >0 data for '{label}'", transform=ax.transAxes, fontsize=8, color='gray', ha='center')
    
    ax.set_title("Histogram of Foot Contact Forces", fontsize=16)
    ax.set_xlabel("Force (N)")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right', fontsize=8)
    
    fig.tight_layout()
    
    pdf_path = os.path.join(output_dir, "foot_contact_forces", "hist_contact_forces_overview.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, dpi=600)
    
    pickle_path = os.path.join(pickle_dir, "hist_contact_forces_overview.pickle")
    if pickle_dir != "":
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)

def _plot_joint_metric(metric_name, data_arr, sim_times, joint_names, leg_row, leg_col, foot_from_joint, contact_state_array, reset_times, constraint_bounds, metric_to_constraint_term_mapping, metric_to_unit_mapping, output_dir, pickle_dir, linewidth):
    fig, axes = plt.subplots(4, 3, sharex=True, figsize=(18, 12))
    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        if row is None or col is None:
            raise ValueError("Could not determine joint row/col for plotting")
        ax = axes[row, col]
        ax.grid()
        ax.plot(sim_times, data_arr[:, j], linewidth=linewidth)
        if metric_name == 'position':
            draw_limits(ax, jn, constraint_bounds)
        else:
            draw_limits(ax, metric_to_constraint_term_mapping[metric_name], constraint_bounds)
        draw_resets(ax, reset_times)

        fid = foot_from_joint[j]
        if fid is not None:
            for stance_start, stance_end in compute_stance_segments(contact_state_array[:, fid].astype(bool)):
                ax.axvspan(sim_times[stance_start], sim_times[stance_end-1], facecolor='gray', alpha=0.5)
        ax.set_title(f"Joint {metric_name.replace('_', ' ')} for {jn}", fontsize=16)
        ax.set_ylabel(f"{metric_name.capitalize()} ({metric_to_unit_mapping[metric_name]})")

    axes[-1, 0].set_xlabel('Time (s)')
    fig.tight_layout()
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f'joint_{metric_name}_grid.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'joint_{metric_name}_grid.pickle'), 'wb') as f:
            pickle.dump(fig, f)

    # overview
    fig_ov, ax = plt.subplots(figsize=(12, 6))
    for j in range(data_arr.shape[1]):
        ax.grid()
        ax.plot(sim_times, data_arr[:, j], label=joint_names[j], linewidth=linewidth, linestyle=get_leg_linestyle(joint_names[j]))
    if metric_name == 'position':
        for jn in joint_names:
            draw_limits(ax, jn, constraint_bounds)
    else:
        draw_limits(ax, metric_to_constraint_term_mapping[metric_name], constraint_bounds)
    draw_resets(ax, reset_times)
    ax.set_xlabel('Time (s)')
    ax.set_title(f"Joint {metric_name.replace('_', ' ')} overview", fontsize=16)
    ax.set_ylabel(f"{metric_name.capitalize()} ({metric_to_unit_mapping[metric_name]})")
    ax.legend(loc='upper right', ncol=2)
    fig_ov.tight_layout()
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f'joint_{metric_name}_overview.pdf')
    fig_ov.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'joint_{metric_name}_overview.pickle'), 'wb') as f:
            pickle.dump(fig_ov, f)

def _plot_hist_joint_grid(metric_name, data_arr, joint_names, leg_row, leg_col, metric_to_unit_mapping, output_dir, pickle_dir):
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle(f"Histogram of Joint {metric_name.replace('_', ' ').title()}", fontsize=18)
    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        ax = axes[row, col]
        ax.grid()

        counts, edges = compute_trimmed_histogram_data(data_arr[:, j])
        # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
        ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
        ax.set_title(jn, fontsize=12)
        ax.set_xlabel(f"{metric_name.capitalize()} ({metric_to_unit_mapping[metric_name]})")
        ax.set_ylabel("Count")
    fig.tight_layout()
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"hist_joint_{metric_name}_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"hist_joint_{metric_name}_grid.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_hist_joint_metric(metric_name, data_arr, joint_names, metric_to_unit_mapping, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for j, jn in enumerate(joint_names):
        ax.grid()
        counts, edges = compute_trimmed_histogram_data(data_arr[:, j])
        # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
        ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0, label=jn)
    ax.set_title(f"Histogram of joint {metric_name.replace('_',' ')}", fontsize=16)
    ax.set_xlabel(f"{metric_name.capitalize()} ({metric_to_unit_mapping[metric_name]})")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"hist_joint_{metric_name}_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"hist_joint_{metric_name}_overview.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_hist_air_time_per_foot_grid(air_segments_per_foot, foot_labels, output_dir, pickle_dir, FIGSIZE):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("Histogram of Air-Time Durations per Foot", fontsize=18)
    for i, label in enumerate(foot_labels):
        durations = air_segments_per_foot[label]
        ax = axes.flat[i]
        ax.grid()
        if durations:
            counts, edges = compute_trimmed_histogram_data(np.array(durations))
            # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
            ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Air Time (s)")
        ax.set_ylabel("Count")
    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", "air_time", "hist_air_time_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "hist_air_time_grid.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_hist_air_time_per_foot_single(label, durations, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if durations:
        ax.grid()
        counts, edges = compute_trimmed_histogram_data(np.array(durations))
        # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
        ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
    ax.set_title(f"Histogram of Air-Time Durations ({label})", fontsize=16)
    ax.set_xlabel("Air Time (s)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    safe_label = label.replace(' ', '_')
    pdf = os.path.join(output_dir, "aggregates", "air_time", f"hist_air_time_{safe_label}.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"hist_air_time_{safe_label}.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_combined_energy(sim_times, combined_energy, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, combined_energy, label='total_energy', linewidth=linewidth)
    ax.grid()
    draw_resets(ax, reset_times)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Total Cumulative Joint Energy', fontsize=16)
    ax.legend()
    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", 'combined_energy_overview.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'combined_energy_overview.pickle'), 'wb') as f:
            pickle.dump(fig, f)

def _plot_reward_time_series(sim_times, reward_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, reward_array, label='Reward', linewidth=linewidth)
    ax.grid()
    draw_resets(ax, reset_times)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reward (-)')
    ax.set_title('Reward at each time step', fontsize=16)
    ax.legend()
    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", 'reward_time_series.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'reward_time_series.pickle'), 'wb') as f:
            pickle.dump(fig, f)

def _plot_cumulative_reward(sim_times, reward_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, np.cumsum(reward_array), label='Cumulative Reward', linewidth=linewidth)
    ax.grid()
    draw_resets(ax, reset_times)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative reward (-)')
    ax.set_title('Cumulative reward', fontsize=16)
    ax.legend()
    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", 'cumulative_reward.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'cumulative_reward.pickle'), 'wb') as f:
            pickle.dump(fig, f)

def _plot_cost_of_transport(sim_times, cost_of_transport_time_series, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, cost_of_transport_time_series, label='cost_of_transport', linewidth=linewidth)
    draw_resets(ax, reset_times)
    ax.grid()
    # Plot running average
    # window_sizes = [25, 50, 300]
    window_sizes = [100]
    for window_size in window_sizes:
        window = np.ones(window_size) / window_size
        running_average = np.convolve(cost_of_transport_time_series, window, mode='valid')
        # Align the running average times. For 'valid', the i-th averaged point corresponds to sim_times[i + (window_size-1)], i.e. the right edge of the window.
        average_times = sim_times[(window_size - 1):]
        ax.plot(average_times, running_average, label=f'{window_size}-sample running avg', linewidth=2.0, linestyle="dashed")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cost of Transport (-)')
    ax.set_title('Instantaneous Cost of Transport', fontsize=16)
    ax.set_ylim(0, 6)
    ax.legend()
    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", "cost_of_transport", 'cost_of_transport_over_time.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'cost_of_transport_over_time.pickle'), 'wb') as f:
            pickle.dump(fig, f)

def _plot_hist_cost_of_transport(cost_of_transport_time_series, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts, edges = compute_trimmed_histogram_data(cost_of_transport_time_series[~np.isnan(cost_of_transport_time_series)])
    # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
    ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
    ax.grid()
    ax.set_xlabel('Cost of Transport (-)')
    ax.set_xlim(0, 6)
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Cost of Transport', fontsize=16)
    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", "cost_of_transport", 'hist_cot_overview.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'hist_cot_overview.pickle'), 'wb') as f:
            pickle.dump(fig, f)

def _plot_combined_base_position(sim_times, base_position_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig_bp, axes_bp = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, axis_label in enumerate(['X', 'Y', 'Z']):
        axes_bp[i].plot(sim_times, base_position_array[:, i], label=f'position_{axis_label}', linewidth=linewidth)
        axes_bp[i].grid()
        draw_resets(axes_bp[i], reset_times)
        axes_bp[i].set_title(f'World Position {axis_label}')
        axes_bp[i].set_ylabel('Position (m)')
        axes_bp[i].legend()
        axes_bp[i].grid(True)
    axes_bp[-1].set_xlabel('Time (s)')
    fig_bp.tight_layout()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_position_subplots_world.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_bp.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'base_position_subplots_world.pickle'), 'wb') as f:
            pickle.dump(fig_bp, f)

def _plot_combined_orientation(sim_times, base_orientation_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig_bo, axes_bo = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, orient_label in enumerate(['Yaw', 'Pitch', 'Roll']):
        axes_bo[i].plot(sim_times, base_orientation_array[:, i], label=orient_label, linewidth=linewidth)
        axes_bo[i].grid()
        draw_resets(axes_bo[i], reset_times)
        axes_bo[i].set_ylabel(f'{orient_label} (rad)')
        axes_bo[i].set_title(f'World Orientation {orient_label}')
        axes_bo[i].legend()
        axes_bo[i].grid(True)
    axes_bo[-1].set_xlabel('Time (s)')
    fig_bo.tight_layout()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_orientation_subplots.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_bo.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'base_orientation_subplots.pickle'), 'wb') as f:
            pickle.dump(fig_bo, f)

def _plot_combined_base_velocity(sim_times, base_linear_velocity_array, commanded_velocity_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig_blv, axes_blv = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, vel_label in enumerate(['Velocity X', 'Velocity Y', 'Velocity Z']):
        axes_blv[i].plot(sim_times, base_linear_velocity_array[:, i], label=vel_label, linewidth=linewidth)
        axes_blv[i].grid()
        if i != 2:
            axes_blv[i].plot(sim_times, commanded_velocity_array[:, i], linestyle='--', label=f'cmd_{vel_label}', linewidth=linewidth)
        draw_resets(axes_blv[i], reset_times)
        axes_blv[i].set_ylabel(f'{vel_label} (m * sec^(-1))')
        axes_blv[i].set_title(f'Base Linear {vel_label}')
        axes_blv[i].legend()
        axes_blv[i].grid(True)
    axes_blv[-1].set_xlabel('Time (s)')
    fig_blv.tight_layout()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_linear_velocity_subplots.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_blv.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'base_linear_velocity_subplots.pickle'), 'wb') as f:
            pickle.dump(fig_blv, f)

def _plot_combined_base_angular_velocities(sim_times, base_angular_velocity_array, commanded_velocity_array, reset_times, output_dir, pickle_dir, FIGSIZE, linewidth):
    fig_bav, axes_bav = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, vel_label in enumerate(['Omega X', 'Omega Y', 'Omega Z']):
        axes_bav[i].plot(sim_times, base_angular_velocity_array[:, i], label=vel_label, linewidth=linewidth)
        axes_bav[i].grid()
        if i == 2:
            axes_bav[i].plot(sim_times, commanded_velocity_array[:, i], linestyle='--', label=f'cmd_{vel_label}', linewidth=linewidth)
        draw_resets(axes_bav[i], reset_times)
        axes_bav[i].set_ylabel(f'{vel_label} (rad * sec^(-1))')
        axes_bav[i].set_title(f'Base {vel_label}')
        axes_bav[i].legend()
        axes_bav[i].grid(True)
    axes_bav[-1].set_xlabel('Time (s)')
    fig_bav.tight_layout()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_angular_velocity_subplots.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_bav.savefig(pdf, dpi=600)
    if pickle_dir != "":
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
            ax.grid()
            draw_resets(ax, reset_times)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)
    fig_overview.tight_layout()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_overview_world.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_overview.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'base_overview_world.pickle'), 'wb') as f:
            pickle.dump(fig_overview, f)

def _plot_gait_diagram(contact_state_array, sim_times, reset_times, foot_labels, output_dir, pickle_dir):
    fig = plot_gait_diagram(contact_state_array, sim_times, reset_times, foot_labels, os.path.join(output_dir, "aggregates", 'gait_diagram.pdf'), spacing=1.0)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'gait_diagram.pickle'), 'wb') as f:
            pickle.dump(fig, f)

# ----------------------------------------------------------------------------------------------------------------------
# box-plot helpers
# ----------------------------------------------------------------------------------------------------------------------

def _plot_box_joint_grid(metric_name, data_arr, joint_names, leg_row, leg_col, metric_to_unit_mapping, output_dir, pickle_dir):
    """
    4x3 grid of box plots, one per joint, for the given metric.
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle(f"box Plot of Joint {metric_name.replace('_', ' ').title()}", fontsize=18)

    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        ax = axes[row, col]
        ax.grid()
        ax.boxplot(x=data_arr[:, j], showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(jn, fontsize=12)
        ax.set_xlabel(f"{metric_name.capitalize()} ({metric_to_unit_mapping[metric_name]})", fontsize=8)

    fig.tight_layout()
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"box_joint_{metric_name}_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"box_joint_{metric_name}_grid.pickle"), "wb") as f:
            pickle.dump(fig, f)


def _plot_box_joint_metric(metric_name, data_arr, joint_names, metric_to_unit_mapping, output_dir, pickle_dir, FIGSIZE):
    """
    Overview box plot (all joints on one axis) for the given metric.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.boxplot(x=[data_arr[:, j] for j in range(data_arr.shape[1])], positions=np.arange(1, len(joint_names) + 1), showmeans=True, showcaps=True, showbox=True, showfliers=False)
    ax.grid()
    ax.set_title(f"Box Plot of Joint {metric_name.replace('_', ' ')}", fontsize=16)
    ax.set_xlabel("Joint")
    ax.set_ylabel(f"{metric_name.capitalize()} ({metric_to_unit_mapping[metric_name]})")
    ax.set_xticks(np.arange(1, len(joint_names) + 1))
    ax.set_xticklabels(joint_names, rotation=45, ha="right")
    fig.tight_layout()

    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"box_joint_{metric_name}_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
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
        ax.grid()
        forces = contact_forces_array[:, i]
        positive = forces[forces > 0]
        if positive.size == 0:
            ax.text(0.5, 0.5, "No forces > 0 N", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="red")
        else:
            ax.boxplot(positive, showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Force (N)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf = os.path.join(output_dir, "foot_contact_forces", "box_contact_forces_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
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
        ax.grid()
        ax.boxplot(data, positions=np.arange(1, len(data) + 1), showmeans=True, showcaps=True, showbox=True, showfliers=False)
    ax.set_title("Box Plot of Foot Contact Forces", fontsize=16)
    ax.set_xlabel("Foot")
    ax.set_ylabel("Force (N)")
    ax.set_xticks(np.arange(1, len(used_labels) + 1))
    ax.set_xticklabels(used_labels)
    fig.tight_layout()

    pdf = os.path.join(output_dir, "foot_contact_forces", "box_contact_forces_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
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
        ax.grid()
        if durations:
            ax.boxplot(durations, showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Air Time (s)")

    fig.tight_layout()
    pdf = os.path.join(output_dir, "aggregates", "air_time", "box_air_time_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "box_air_time_grid.pickle"), "wb") as f:
            pickle.dump(fig, f)


def _plot_box_air_time_per_foot_single(label, durations, output_dir, pickle_dir, FIGSIZE):
    """
    Single-foot box plot of air-time durations.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if durations:
        ax.boxplot(durations, showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.grid()
    else:
        ax.text(0.5, 0.5, "No air-time segments", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="red")

    ax.set_title(f"Box Plot of Air-Time Durations ({label})", fontsize=16)
    ax.set_xlabel("Air Time (s)")
    fig.tight_layout()

    safe_label = label.replace(" ", "_")
    pdf = os.path.join(output_dir, "aggregates", "air_time", f"box_air_time_{safe_label}.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
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
        ax.grid()
    ax.set_xlabel("Cost of Transport")
    ax.set_title("Box Plot of Cost of Transport", fontsize=16)
    fig.tight_layout()

    pdf = os.path.join(output_dir, "aggregates", "cost_of_transport", "box_cot_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "box_cot_overview.pickle"), "wb") as f:
            pickle.dump(fig, f)

# ----------------------------------------------------------------------------------------------------------------------
# Main plot generation orchestrator
# ----------------------------------------------------------------------------------------------------------------------

def generate_plots(data, output_dir, interactive=False):
    """
    Recreate all plots from loaded data.
    - data: numpy.lib.npyio.NpzFile containing arrays
    - metrics: dict loaded from metrics_summary.json
    """
    os.makedirs(output_dir, exist_ok=True)
    # pickle_dir = os.path.join(output_dir, "plot_figures_serialized")
    pickle_dir = "" # Takes up too much space and can be regenerated with this script anyway (sim_data.npz is saved)
    if pickle_dir != "":
        os.makedirs(pickle_dir, exist_ok=True)

    start_time = time.time()

    linewidth = 1
    FIGSIZE = (16, 9)

    # Unpack arrays
    sim_times = data['sim_times']
    reset_times = data['reset_times'].tolist()
    step_dt = sim_times[1] - sim_times[0] # Important: step_dt != sim_dt due to decimation
    # Subtract sim_times[0] because data might be sliced, so sim_times[0] is the t0. Plots will use absolute time as x axis, but reset_timesteps need to be within slice.
    reset_timesteps = [int(round((reset_time - sim_times[0]) / step_dt)) for reset_time in reset_times]
    contact_forces_array = data['contact_forces_array']
    foot_labels = list(data['foot_labels'])
    contact_state_array = data['contact_state_array']
    constraint_bounds = data['constraint_bounds'].item()
    joint_names = list(data['joint_names'])
    total_robot_mass = data['total_robot_mass']
    power_array = data["power_array"]
    reward_array = data["reward_array"]
    joint_positions = data['joint_positions_array']
    joint_velocities = data['joint_velocities_array']
    joint_accelerations = data['joint_accelerations_array']
    joint_torques = data['joint_torques_array']
    joint_action_rates = data['action_rate_array']
    base_positions = data['base_position_array']
    base_orientations = data['base_orientation_array']
    base_linear_velocities = data['base_linear_velocity_array']
    base_angular_velocities = data['base_angular_velocity_array']
    base_commanded_velocities = data['commanded_velocity_array']
    energy_per_joint, combined_energy, cost_of_transport_time_series = compute_energy_arrays(power_array=power_array, base_lin_vel=base_linear_velocities, reset_steps=reset_timesteps, step_dt=step_dt, robot_mass=total_robot_mass)
    foot_positions_body_frame    = data['foot_positions_body_frame_array'] # (T,4,3)
    foot_positions_contact_frame = data['foot_positions_contact_frame_array'] # (T,4,3)
    foot_positions_world_frame   = data['foot_positions_world_frame_array'] # (T,4,3)
    foot_heights_body_frame = foot_positions_body_frame[:, :, 2]
    foot_heights_contact_frame = foot_positions_contact_frame[:, :, 2]
    step_heights = compute_swing_heights(contact_state=contact_state_array, foot_heights_contact=foot_heights_contact_frame, reset_steps=reset_timesteps, foot_labels=foot_labels)
    step_lengths = compute_swing_lengths(contact_state=contact_state_array, foot_positions_world=foot_positions_world_frame, reset_steps=reset_timesteps, foot_labels=foot_labels)
    swing_durations = compute_swing_durations(contact_state=contact_state_array, sim_env_step_dt=step_dt, foot_labels=foot_labels)

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
        'position': joint_positions,
        'velocity': joint_velocities,
        'acceleration': joint_accelerations,
        'torque': joint_torques,
        'action_rate': joint_action_rates,
        'energy': energy_per_joint,
        'power': power_array,
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
        futures.append(executor.submit(_animate_body_frame_foot_positions, foot_positions_body_frame, contact_state_array, foot_labels, os.path.join(output_dir, "foot_positions_body_frame", "foot_positions_body_frame_animation.mp4"), fps=20))
        
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
                swing_durations, foot_labels, output_dir, pickle_dir, FIGSIZE
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

        futures.append(
            executor.submit(
                _plot_reward_time_series,
                sim_times, reward_array, reset_times,
                output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )

        futures.append(
            executor.submit(
                _plot_cumulative_reward,
                sim_times, reward_array, reset_times,
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
        for label, durations in swing_durations.items():
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
                sim_times, base_positions, reset_times,
                output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )
        futures.append(
            executor.submit(
                _plot_combined_orientation,
                sim_times, base_orientations, reset_times,
                output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )
        futures.append(
            executor.submit(
                _plot_combined_base_velocity,
                sim_times, base_linear_velocities, base_commanded_velocities,
                reset_times, output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )
        futures.append(
            executor.submit(
                _plot_combined_base_angular_velocities,
                sim_times, base_angular_velocities, base_commanded_velocities,
                reset_times, output_dir, pickle_dir, FIGSIZE, linewidth
            )
        )
        futures.append(
            executor.submit(
                _plot_total_base_overview,
                sim_times, base_positions, base_orientations,
                base_linear_velocities, base_angular_velocities,
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
                swing_durations, foot_labels, output_dir, pickle_dir, FIGSIZE
            )
        )
        for label, durations in swing_durations.items():
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

        # ---------------- Foot-height time series ----------------
        positions_axes = {'X': 0, 'Y': 1, 'Z': 2}
        for axis_label, axis_idx in positions_axes.items():
            positions_axis = foot_positions_body_frame[:, :, axis_idx]
            metric_dict    = _array_to_metric_dict(positions_axis, foot_labels)
            subdir = os.path.join("foot_positions_body_frame", f"{axis_label}")

            futures.append(
                executor.submit(
                    _plot_foot_position_time_series,
                    positions_axis, axis_label, 'body_frame',
                    sim_times, foot_labels, contact_state_array, reset_times,
                    output_dir, pickle_dir, FIGSIZE, linewidth, subdir
                )
            )

            futures.append(
                executor.submit(
                    _plot_hist_metric_grid,
                    metric_dict,
                    f"Histogram of Foot {axis_label} Position (body frame, CoM)",
                    f"{axis_label} (m)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )
            futures.append(
                executor.submit(
                    _plot_hist_metric_overview,
                    metric_dict,
                    f"Histogram of Foot {axis_label} Position (body frame, CoM) overview",
                    f"{axis_label} (m)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )

            futures.append(
                executor.submit(
                    _plot_box_metric_grid,
                    metric_dict,
                    f"Box Plot of Foot {axis_label} Position (body frame, CoM)",
                    f"{axis_label} (m)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )
            futures.append(
                executor.submit(
                    _plot_box_metric_overview,
                    metric_dict,
                    f"Box Plot of Foot {axis_label} Position (body frame, CoM) overview",
                    f"{axis_label} (m)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )

        # Same for contact frame
        for axis_label, axis_idx in positions_axes.items():
            positions_axis = foot_positions_contact_frame[:, :, axis_idx]
            metric_dict    = _array_to_metric_dict(positions_axis, foot_labels)
            subdir = os.path.join("foot_positions_contact_frame", f"{axis_label}")

            futures.append(
                executor.submit(
                    _plot_foot_position_time_series,
                    positions_axis, axis_label, 'contact_frame',
                    sim_times, foot_labels, contact_state_array, reset_times,
                    output_dir, pickle_dir, FIGSIZE, linewidth, subdir
                )
            )

            futures.append(
                executor.submit(
                    _plot_hist_metric_grid,
                    metric_dict,
                    f"Histogram of Foot {axis_label} Position (contact frame, toe tip)",
                    f"{axis_label} (m)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )
            futures.append(
                executor.submit(
                    _plot_hist_metric_overview,
                    metric_dict,
                    f"Histogram of Foot {axis_label} Position (contact frame, toe tip) overview",
                    f"{axis_label} (m)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )

            futures.append(
                executor.submit(
                    _plot_box_metric_grid,
                    metric_dict,
                    f"Box Plot of Foot {axis_label} Position (contact frame, toe tip)",
                    f"{axis_label} (m)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )
            futures.append(
                executor.submit(
                    _plot_box_metric_overview,
                    metric_dict,
                    f"Box Plot of Foot {axis_label} Position (contact frame, toe tip) overview",
                    f"{axis_label} (m)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )

        # ---------------- Max step-height ----------------
        futures.append(
            executor.submit(
                _plot_hist_metric_grid,
                step_heights,
                "Histogram of Step Height (contact frame)",
                "Height (m)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_height", FIGSIZE=FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_hist_metric_overview,
                step_heights,
                "Histogram of Step Height (overview)",
                "Height (m)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_height", FIGSIZE=FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_box_metric_grid,
                step_heights,
                "Box Plot of Step Height",
                "Height (m)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_height", FIGSIZE=FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_box_metric_overview,
                step_heights,
                "Box Plot of Step Height (overview)",
                "Height (m)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_height", FIGSIZE=FIGSIZE
            )
        )

        # ---------------- Step length ----------------
        futures.append(
            executor.submit(
                _plot_hist_metric_grid,
                step_lengths,
                "Histogram of Step Length",
                "Step Length (m)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_length", FIGSIZE=FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_hist_metric_overview,
                step_lengths,
                "Histogram of Step Length (overview)",
                "Step Length (m)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_length", FIGSIZE=FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_box_metric_grid,
                step_lengths,
                "Box Plot of Step Length",
                "Step Length (m)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_length", FIGSIZE=FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_box_metric_overview,
                step_lengths,
                "Box Plot of Step Length (overview)",
                "Step Length (m)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_length", FIGSIZE=FIGSIZE
            )
        )

        foot_heatmap_bin_size = 50 # 2.5cmx2.5cm

        futures.append(
            executor.submit(
                _plot_body_frame_foot_position_heatmap_grid,
                foot_positions_body_frame,
                foot_labels,
                output_dir,
                pickle_dir,
                bin_count=foot_heatmap_bin_size,
                FIGSIZE=(20, 20),
            )
        )

        for idx, lbl in enumerate(foot_labels):
            futures.append(
                executor.submit(
                    _plot_body_frame_foot_position_heatmap_single,
                    foot_positions_body_frame,
                    idx,
                    lbl,
                    output_dir,
                    pickle_dir,
                    bin_count=foot_heatmap_bin_size,
                    FIGSIZE=(20, 20),
                )
            )

        futures.append(
            executor.submit(
                _plot_body_frame_foot_position_heatmap,
                foot_positions_body_frame,
                output_dir,
                pickle_dir,
                bin_count=foot_heatmap_bin_size,
                FIGSIZE=(20, 20),
            )
        )

        # Ensure all tasks complete
        for f in futures:
            f.result()

    end_time = time.time()
    print(f"Plot generation took {(end_time-start_time):.4f} seconds.")

    # Interactive display if requested
    if interactive:
        plt.ion()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from saved simulation data.")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the .npz file containing recorded sim data.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Desired plot output path. Will be created if nonexistant.")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Show figures interactively in addition to saving.")
    parser.add_argument("--start_step", type=int, default=None,
                    help="(Optional) first step to include (global index, 0-based)")
    parser.add_argument("--end_step", type=int, default=None,
                    help="(Optional) last step to include, inclusive")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.data_file)
    if args.start_step is not None:
        s = slice(args.start_step, args.end_step + 1)
        data = {k: (v[s] if (k.endswith("_array") or k == "sim_times") else v) for k, v in data.items()}

        # bring resets in-range and shift them so that t=0 is the first
        sim_times = data["sim_times"]
        t0, t1 = float(sim_times[0]), float(sim_times[-1])
        print(t0, t1)
        full_resets = data["reset_times"]
        in_window = (full_resets >= t0) & (full_resets <= t1)
        data["reset_times"] = full_resets[in_window]
        print(data["reset_times"])
        
    generate_plots(data, args.output_dir, interactive=args.interactive)

if __name__ == "__main__":
    main()