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
import matplotlib.colors as colors
import scienceplots
import time
import queue
import warnings
import subprocess
import io
import shutil
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count
from matplotlib.axes import Axes
from functools import partialmethod
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use(['science','ieee'])

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.constrained_layout.use': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.grid': True,
    'grid.linestyle': '--',
})

# Patch to  minor plots
Axes.grid = partialmethod(Axes.grid, which='both') # type: ignore

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

def plot_gait_diagram(contact_states: np.ndarray, sim_times: np.ndarray, reset_times: list[float], foot_labels: list[str], output_path: str, spacing: float = 1.0):
    T, F = contact_states.shape
    assert sim_times.shape[0] == T, "sim_times length must match contact_states"

    fig, ax = plt.subplots(figsize=(180 if sim_times[0] == 0.0 else 24, F * 1.2))
    ax.set_xlabel(r'Time ($\text{s}$)')
    ax.set_title('Gait Diagram with Air and Contact Times (white text = contact/stance phase, black text = air/swing phase)', fontsize=18)

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
            ax.text(t_mid, y_text, f"{duration:.3f}s", ha='center', va='center', color='white', fontsize=8, rotation=90)

        swing_segments = compute_swing_segments(in_contact)

        # Annotate durations
        for a, b in swing_segments:
            t_start = sim_times[a]
            t_end   = sim_times[b - 1]
            duration = t_end - t_start
            t_mid = 0.5 * (t_start + t_end)
            ax.text(t_mid, y0 + spacing * 0.5, f"{duration:.3f} s", ha='center', va='center', fontsize=8, rotation=90)

    ax.set_xticks(np.arange(0, sim_times[-1], 1))
    ax.set_yticks([i * spacing for i in range(F)])
    ax.set_yticklabels(foot_labels)
    ax.margins(x=0.005)
    ax.set_ylim(-spacing * 0.5, (F - 1) * spacing + spacing)
    ax.legend(loc='upper right', ncol=1)
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

def _plot_body_frame_foot_position_heatmap(foot_positions_body_frame: np.ndarray, output_dir: str, pickle_dir: str, gridsize: int = 100, FIGSIZE: tuple[int, int] = (20, 20)):
    """
    Discretised hexbin plot of all foot XY positions in the *body* frame.
    """
    xy = foot_positions_body_frame[:, :, :2].reshape(-1, 2) # (N, 2)
    x, y = xy[:, 0], xy[:, 1]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    ax.set_title("Foot-position occupancy hexbin (body frame, CoM, top-down)", fontsize=28)
    if len(x) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap='viridis', mincnt=1)
        counts = hb.get_array()
        
        cbar_label = "Occupancy (samples)"
        if counts.size > 0 and counts.max() > 10 and (counts.max() / counts.min() > 20):
            hb.set_norm(colors.LogNorm(vmin=counts.min(), vmax=counts.max()))
            cbar_label = r"$\log_{10}(\text{Occupancy})$"

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label(cbar_label, size=22)
        cb.ax.tick_params(labelsize=20)

        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r"Body-X ($\text{m}$)", fontsize=22)
        ax.set_ylabel(r"Body-Y ($\text{m}$)", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)

    pdf_dir = os.path.join(output_dir, "foot_com_positions_body_frame", "heatmap")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "foot_com_position_heatmap_body_frame.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "foot_com_position_heatmap_body_frame.pickle"), "wb") as f:
            pickle.dump(fig, f)
    plt.close(fig)

def _plot_body_frame_foot_position_heatmap_grid(foot_positions_body_frame: np.ndarray, foot_labels: list[str], output_dir: str, pickle_dir: str, gridsize: int = 100, FIGSIZE: tuple[int, int] = (20, 20)):
    """
    2x2 grid of occupancy hexbin plots - one per foot - in body frame.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("Foot-position Hexbin Plots (body frame, per foot)", fontsize=30)

    for i, (ax, lbl) in enumerate(zip(axes.flat, foot_labels)):
        xy = foot_positions_body_frame[:, i, :2] # (T, 2)
        x, y = xy[:, 0], xy[:, 1]

        ax.set_title(lbl, fontsize=26)
        if len(x) == 0: # No data
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap='viridis', mincnt=1)
        
        counts = hb.get_array()
        cbar_label = "Occupancy (samples)"
        if counts.size > 0 and counts.max() > 10 and (counts.max() / counts.min() > 20):
             hb.set_norm(colors.LogNorm(vmin=max(1, counts.min()), vmax=counts.max()))
             cbar_label = r"$\log_{10}(\text{Occupancy})$"

        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r"Body-X ($\text{m}$)", fontsize=22)
        ax.set_ylabel(r"Body-Y ($\text{m}$)", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label(cbar_label, size=22)
        cb.ax.tick_params(labelsize=20)

    # fig.subplots_adjust(top=0.92, hspace=0.4)
    pdf_dir = os.path.join(output_dir, "foot_com_positions_body_frame", "heatmap")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "foot_com_position_heatmap_body_frame_grid.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')

    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "foot_com_position_heatmap_body_frame_grid.pickle"), "wb") as f:
            pickle.dump(fig, f)

    plt.close(fig)

def _plot_body_frame_foot_position_heatmap_single(foot_positions_body_frame: np.ndarray, foot_idx: int, foot_label: str, output_dir: str, pickle_dir: str, gridsize: int = 100, FIGSIZE: tuple[int, int] = (20, 20)):
    xy = foot_positions_body_frame[:, foot_idx, :2]
    x, y = xy[:, 0], xy[:, 1]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(f"Foot-position hexbin (body frame) – {foot_label}", fontsize=28)

    if len(x) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap='viridis', mincnt=1)
        counts = hb.get_array()
        
        cbar_label = "Occupancy (samples)"
        if counts.size > 0 and counts.max() > 10 and (counts.max() / counts.min() > 20):
            hb.set_norm(colors.LogNorm(vmin=counts.min(), vmax=counts.max()))
            cbar_label = r"$\log_{10}(\text{Occupancy})$"

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label(cbar_label, size=22)
        cb.ax.tick_params(labelsize=20)
        # ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(r"Body-X ($\text{m}$)", fontsize=22)
    ax.set_ylabel(r"Body-Y ($\text{m}$)", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)

    safe_lbl = foot_label.replace(" ", "_")
    pdf_dir = os.path.join(output_dir, "foot_com_positions_body_frame", "heatmap", safe_lbl.lower())
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"foot_position_heatmap_{safe_lbl.lower()}_com.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')
    
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"foot_com_position_heatmap_{safe_lbl.lower()}.pickle"), "wb") as f:
            pickle.dump(fig, f)

    plt.close(fig)

# Globals for worker processes, initialized by init_animation_worker
_worker_shared_queue = None
_worker_foot_positions = None
_worker_contact_state = None
# Matplotlib objects, also global per worker
_worker_fig = None
_worker_ax = None
_worker_scatters = None
# Other static data
_worker_foot_labels = None
_worker_colours = None
_worker_xlims = None
_worker_ylims = None
_worker_dpi = None


def init_animation_worker(shared_queue, foot_positions, contact_state,
                        foot_labels_data, colours_data, xlims_data, ylims_data, dpi_val):
    """Initializer for each worker process in the Pool. Sets up Matplotlib objects once."""
    global _worker_shared_queue, _worker_foot_positions, _worker_contact_state, \
           _worker_fig, _worker_ax, _worker_scatters, \
           _worker_foot_labels, _worker_colours, _worker_xlims, _worker_ylims, _worker_dpi
    
    _worker_shared_queue = shared_queue
    _worker_foot_positions = foot_positions
    _worker_contact_state = contact_state
    _worker_foot_labels = foot_labels_data
    _worker_colours = colours_data
    _worker_xlims = xlims_data
    _worker_ylims = ylims_data
    _worker_dpi = dpi_val

    # --- Setup the plot objects ONCE per worker ---
    _worker_fig, _worker_ax = plt.subplots(figsize=(8, 6)) # figsize can be parameterized if needed
    # _worker_ax.set_aspect('equal')
    _worker_ax.set_xlabel('Body-X (m)')
    _worker_ax.set_ylabel('Body-Y (m)')
    _worker_scatters = [_worker_ax.scatter([], [], s=60, c=c, edgecolor=c, label=lbl)
                        for c, lbl in zip(_worker_colours, _worker_foot_labels)]
    _worker_ax.legend(loc='upper right')
    _worker_ax.set_xlim(_worker_xlims)
    _worker_ax.set_ylim(_worker_ylims)
    
    # Apply tight_layout once during setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _worker_fig.tight_layout()

def generate_frame_to_buffer(k: int): # Frame index k is the only argument
    """
    Generates a single frame and puts its raw byte data into a shared queue.
    The frame index `k` is the only argument.
    """
    global _worker_shared_queue, _worker_foot_positions, _worker_contact_state, \
           _worker_fig, _worker_ax, _worker_scatters, \
           _worker_foot_labels, _worker_colours, _worker_dpi # xlims, ylims are set on ax already
    
    # Assert that globals have been initialized
    assert _worker_shared_queue is not None, "Worker shared_queue not initialized"
    assert _worker_foot_positions is not None, "Worker foot_positions not initialized"
    assert _worker_contact_state is not None, "Worker contact_state not initialized"
    assert _worker_fig is not None, "Worker Matplotlib figure not initialized"
    assert _worker_ax is not None, "Worker Matplotlib axes not initialized"
    assert _worker_scatters is not None, "Worker Matplotlib scatter artists not initialized"
    assert _worker_foot_labels is not None, "Worker foot_labels not initialized"
    assert _worker_colours is not None, "Worker colours not initialized"
    assert _worker_dpi is not None, "Worker dpi not initialized"

    try:
        # Access data from globals
        foot_positions_body_frame = _worker_foot_positions
        contact_state_array = _worker_contact_state
        # foot_labels = _worker_foot_labels # Used in init
        colours = _worker_colours
        # xlims = _worker_xlims # Used in init
        # ylims = _worker_ylims # Used in init
        dpi = _worker_dpi
        
        fig = _worker_fig
        ax = _worker_ax
        scatters = _worker_scatters

        # --- Update dynamic parts of the plot for frame k ---
        ax.set_title(f'Foot Trajectories (Body Frame) - Frame {k}') # Update title

        for i, sc in enumerate(scatters):
            xy = foot_positions_body_frame[k, i, :2].reshape(1, 2)
            sc.set_offsets(xy)
            if contact_state_array[k, i]:
                sc.set_facecolor(colours[i])
            else:
                sc.set_facecolor('none')
            # sc.set_edgecolor(colours[i]) # Edge color is static, set in init

        # --- Save the figure to an in-memory buffer ---
        with io.BytesIO() as buf:
            fig.savefig(buf, format='png', dpi=dpi)
            buf.seek(0)
            frame_bytes = buf.getvalue()

        # DO NOT call plt.close(fig) here, as the figure is reused.
        # It's cleaned up when the worker process terminates.

        # Put the frame index and its data onto the shared queue
        _worker_shared_queue.put((k, frame_bytes))

    except Exception as e:
        # Ensure errors in workers are reported
        print(f"Error in worker for frame {k}: {e}")
        _worker_shared_queue.put((k, None)) # Signal failure for this frame


def _animate_body_frame_pipe_to_ffmpeg(foot_positions_body_frame: np.ndarray, contact_state_array: np.ndarray, foot_labels: list[str], output_path: str, fps: int = 30, dpi: int = 300):
    """
    Maximally efficient animation generation using a producer-consumer pipeline:
    1. Parallel frame rendering to in-memory buffers (producers).
    2. A single FFmpeg process consumes frames via stdin (consumer).
    3. GPU-accelerated H.265 (HEVC) encoding with FFmpeg (NVENC).
    """
    T = foot_positions_body_frame.shape[0]
    colours = ['red', 'blue', 'green', 'purple']

    # --- 1. Start the FFmpeg Consumer Process ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_time = time.time()
    
    # FFmpeg command to read PNG image data from stdin
    ffmpeg_cmd = [
        'ffmpeg',
        '-r', str(fps),                       # Frame rate of the input stream
        '-f', 'image2pipe',                   # Tell FFmpeg to expect a stream of images
        '-vcodec', 'png',                     # The codec of the images in the pipe is PNG
        '-i', '-',                            # The input is stdin ('-')
        '-c:v', 'hevc_nvenc',                 # VIDEO CODEC: H.265 (HEVC) with NVIDIA Acceleration
        '-preset', 'p1',                      # PRESET: p7 is fastest, p1 is best quality
        '-pix_fmt', 'yuv420p',                # Pixel format for compatibility
        '-loglevel', 'info',                 # Suppress verbose frame-by-frame logging to stderr
        '-y',                                 # Overwrite output file if it exists
        output_path
    ]

    # Popen starts the process without blocking, allowing us to feed it data.
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # --- 2. Setup the Producer Pool and Shared Queue ---
    # Use a standard multiprocessing.Queue for better performance
    # Maxsize provides backpressure if producers are too fast
    num_processes = cpu_count()
    # frame_queue = multiprocessing.Queue(maxsize=num_processes * 4)
    # Using Manager queue for now as direct mp.Queue with spawn can be tricky with initializers if not careful
    # Reverting to Manager.Queue for stability, will re-evaluate mp.Queue if this isn't enough.
    # For true mp.Queue, it would need to be created outside and passed to initializer,
    # or workers would need a different way to access it if it's not easily inheritable with 'spawn'.
    # The primary bottleneck is likely data pickling, which worker_init fixes.
    manager = Manager() # Keep Manager for Queue for now, focus on worker_init first.
    frame_queue = manager.Queue(maxsize=num_processes * 4 if num_processes else 4)



    # Pre-calculate rendering data to pass to all workers
    x_min, x_max = np.min(foot_positions_body_frame[:, :, 0]), np.max(foot_positions_body_frame[:, :, 0])
    y_min, y_max = np.min(foot_positions_body_frame[:, :, 1]), np.max(foot_positions_body_frame[:, :, 1])
    padding = 0.05
    # Data for worker initialization
    init_args = (
        frame_queue, # Pass the queue to the workers
        foot_positions_body_frame, contact_state_array,
        foot_labels, colours,
        (x_min - padding, x_max + padding), (y_min - padding, y_max + padding),
        dpi
    )

    print(f"Starting frame generation with {num_processes} worker processes for {T} frames (DPI: {dpi})...")
    
    # Worker function no longer needs data or queue via partial, it gets them from globals
    # The iterable for map_async is just the frame indices
    with Pool(processes=num_processes, initializer=init_animation_worker, initargs=init_args) as pool:
        # Dispatch all jobs asynchronously to the pool. This is non-blocking.
        async_result = pool.map_async(generate_frame_to_buffer, range(T))

        # --- 3. Orchestrator Loop: Re-order and Pipe Frames to FFmpeg ---
        print("Starting to pipe frames to FFmpeg...")
        frames_processed_count = 0 # Counts frames either successfully written or acknowledged as failed
        frames_actually_written_to_pipe = 0
        next_frame_to_write = 0
        reorder_buffer = {} # Holds out-of-order frames (frame_index: frame_bytes | None)

        # Loop until all T frames have been processed (either written or acknowledged as failed)
        while frames_processed_count < T:
            try:
                # Get the next available frame data from the queue
                # Timeout helps detect if workers hang or if queue logic is flawed
                k, frame_bytes = frame_queue.get(timeout=T * 0.5)

                reorder_buffer[k] = frame_bytes # Store frame_bytes (or None if failed)

                # Attempt to write all consecutive frames that are now available
                while next_frame_to_write in reorder_buffer:
                    current_frame_bytes = reorder_buffer.pop(next_frame_to_write)
                    
                    if current_frame_bytes is not None: # Successfully rendered frame
                        try:
                            ffmpeg_process.stdin.write(current_frame_bytes)
                            frames_actually_written_to_pipe +=1
                        except (IOError, BrokenPipeError):
                            print(f"FFmpeg process closed pipe unexpectedly while writing frame {next_frame_to_write}. Aborting.")
                            # Ensure outer loop terminates if FFmpeg crashes
                            frames_processed_count = T
                            break # Break from inner write loop
                    else: # Frame failed to render
                        print(f"Skipping failed frame {next_frame_to_write} (not writing to FFmpeg).")

                    frames_processed_count += 1 # Increment for every frame index handled
                    next_frame_to_write += 1
                
                if frames_processed_count % 100 == 0 and frames_processed_count > 0:
                     print(f"  ... {frames_processed_count} / {T} frames processed ({frames_actually_written_to_pipe} written to FFmpeg).")
            
            except queue.Empty:
                # This might happen if workers are slower than the timeout or if all workers finished
                # but not all frames were accounted for (e.g., due to an earlier error).
                # The async_result.get() below will catch worker errors.
                print("Frame queue is empty. Checking if all workers have completed...")
                if async_result.ready(): # Check if all tasks in map_async are done
                    if frames_processed_count < T:
                        print(f"Warning: Queue empty and workers finished, but only {frames_processed_count}/{T} frames processed.")
                    break # Exit orchestrator loop; rely on async_result.get() for final status
                else:
                    print("Workers still running, continuing to wait for frames...")
                    # Continue waiting, the timeout on queue.get() will trigger again if needed.
            
            if frames_processed_count == T and next_frame_to_write != T:
                 # This case handles if the last frames were failures and reorder_buffer is now empty
                 # but next_frame_to_write hasn't reached T yet.
                 print(f"All {T} frames processed. Some later frames might have failed.")
                 break

        print(f"Orchestrator loop finished. {frames_processed_count}/{T} frames processed. {frames_actually_written_to_pipe} frames written to FFmpeg.")
        # Ensure all worker processes have completed and handle any exceptions from them.
        # This is crucial for robust error handling and proper pool shutdown.
        pool.close() # Signal that no more tasks will be submitted to this pool.
        print("Waiting for all frame generation tasks to complete...")
        try:
            worker_completion_timeout = T * 0.5
            async_result.get(timeout=worker_completion_timeout)
            print("All frame generation tasks completed successfully.")
        except multiprocessing.TimeoutError:
            print(f"Timeout waiting for worker processes to complete after {worker_completion_timeout}s. Some frames may not have been generated.")
        except Exception as e:
            print(f"An error occurred in one of the worker processes: {e}")
        
        pool.join() # Wait for all worker processes to terminate.
        print("Worker pool joined.")

    # --- 4. Finalize and Clean Up FFmpeg ---
    # ffmpeg_process.stdin was written to by the orchestrator loop.
    # We don't need to explicitly close it here if we are using communicate(),
    # as communicate() will handle closing stdin after sending its (optional) input.
    # Since we are passing no new input to communicate(), it should close stdin.

    print("Attempting to communicate with FFmpeg (finalize encoding)...")
    try:
        # Communicate will read remaining stdout/stderr and wait for process termination.
        # Timeout should be sufficient for FFmpeg to finish encoding the already piped frames.
        ffmpeg_timeout = T * 0.2
        _stdout, _stderr = ffmpeg_process.communicate(timeout=ffmpeg_timeout)
    except subprocess.TimeoutExpired:
        ffmpeg_process.kill()
        _stdout, _stderr = ffmpeg_process.communicate()
        print("\n--- FFMPEG TIMED OUT ---")
        print(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        if _stderr:
            print(f"FFmpeg stderr (on timeout):\n{_stderr.decode('utf-8', errors='ignore')}")
        return # Exit if timed out

    ret_code = ffmpeg_process.returncode
    
    # Check for errors
    if ret_code != 0:
        print("\n--- FFMPEG FAILED ---")
        print(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        if _stderr:
            error_output = _stderr.decode('utf-8', errors='ignore')
            print(f"FFmpeg stderr:\n{error_output}")
    else:
        end_time = time.time()
        print(f"\nSuccessfully created animation with avg fps={(T / (end_time - start_time)):.4f} at: {output_path}")
        if _stderr: # FFmpeg can sometimes output warnings to stderr even on success
            error_output = _stderr.decode('utf-8', errors='ignore')
            if error_output.strip():
                print(f"FFmpeg stderr (success with warnings):\n{error_output}")

def _animate_body_frame_foot_positions(foot_positions_body_frame: np.ndarray, contact_state_array: np.ndarray, foot_labels: list[str], output_path: str, fps: int = 30):
    """
    [DEPRECATED] Use _animate_body_frame_pipe_to_ffmpeg for a much faster, parallelized implementation.
    Top-down animation of body-frame foot XY positions.
    - filled marker : stance / contact
    - hollow marker : swing / air
    """
    T = foot_positions_body_frame.shape[0]
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel(r'Body-X ($\text{m}$)')
    ax.set_ylabel(r'Body-Y ($\text{m}$)')
    ax.set_title('Foot trajectories (body frame, CoM, top-down)')

    # create one PathCollection per foot
    scatters = [ax.scatter([], [], s=60, c=colours[i % len(colours)], edgecolor=colours[i % len(colours)], label=lbl) for i, lbl in enumerate(foot_labels)]
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
                sc.set_facecolor(colours[i % len(colours)]) # filled (stance)
            else:
                sc.set_facecolor('none') # hollow (swing)
            sc.set_edgecolor(colours[i % len(colours)]) # always draw edges
        return scatters

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani = animation.FuncAnimation(fig, _animate, init_func=_init, frames=T, interval=1000 / fps, blit=True)
    ani.save(output_path, fps=fps)
    plt.close(fig)


def _plot_body_frame_foot_position_xy_grid(
    foot_positions_body_frame: np.ndarray,
    foot_labels: list[str],
    contact_state_array: np.ndarray,
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (20, 20)
):
    """
    2x2 grid of XY trajectory plots - one per foot - in body frame.
    Stance phases are marked with scatter points.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("Foot XY Trajectories (body frame, per foot)", fontsize=22)

    for i, (ax, lbl) in enumerate(zip(axes.flat, foot_labels)):
        x = foot_positions_body_frame[:, i, 0]
        y = foot_positions_body_frame[:, i, 1]
        
        # Full trajectory path
        ax.plot(x, y, alpha=0.6, label="Swing Trajectory")

        # Stance points
        stance_indices = np.where(contact_state_array[:, i])[0]
        if len(stance_indices) > 0:
            ax.scatter(x[stance_indices], y[stance_indices], s=5, label="Stance Points", color='red')

        ax.set_title(lbl)
        ax.set_xlabel(r"Body-X ($\text{m}$)")
        ax.set_ylabel(r"Body-Y ($\text{m}$)")
        ax.legend()
        ax.grid(True)

    pdf_dir = os.path.join(output_dir, "foot_com_positions_body_frame", "xy_trajectory")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "foot_com_position_xy_trajectory_body_frame_grid.pdf")
    fig.savefig(pdf_path, dpi=600)

    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "foot_com_position_xy_trajectory_body_frame_grid.pickle"), "wb") as f:
            pickle.dump(fig, f)

    plt.close(fig)


def _plot_body_frame_foot_position_xy_overview(
    foot_positions_body_frame: np.ndarray,
    foot_labels: list[str],
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (20, 20)
):
    """
    Overlayed XY trajectory of all feet in the *body* frame.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title("Foot XY Trajectories (body frame, overview)", fontsize=22)

    for i, lbl in enumerate(foot_labels):
        x = foot_positions_body_frame[:, i, 0]
        y = foot_positions_body_frame[:, i, 1]
        ax.plot(x, y, label=lbl, alpha=0.7)

    ax.set_xlabel(r"Body-X ($\text{m}$)")
    ax.set_ylabel(r"Body-Y ($\text{m}$)")
    # ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(True)

    pdf_dir = os.path.join(output_dir, "foot_com_positions_body_frame", "xy_trajectory")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "foot_com_position_xy_trajectory_body_frame_overview.pdf")
    fig.savefig(pdf_path, dpi=600)

    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "foot_com_position_xy_trajectory_body_frame_overview.pickle"), "wb") as f:
            pickle.dump(fig, f)

    plt.close(fig)


def _plot_body_frame_foot_position_xy_single(
    foot_positions_body_frame: np.ndarray,
    foot_idx: int,
    foot_label: str,
    contact_state_array: np.ndarray,
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (20, 20)
):
    """
    XY trajectory plot for a single foot in body frame.
    Stance phases are marked with scatter points.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    x = foot_positions_body_frame[:, foot_idx, 0]
    y = foot_positions_body_frame[:, foot_idx, 1]
    
    # Full trajectory path
    ax.plot(x, y, alpha=0.6, label="Swing Trajectory")

    # Stance points
    stance_indices = np.where(contact_state_array[:, foot_idx])[0]
    if len(stance_indices) > 0:
        ax.scatter(x[stance_indices], y[stance_indices], s=5, label="Stance Points", color='red')

    ax.set_title(f"Foot XY Trajectory (body frame) - {foot_label}")
    ax.set_xlabel(r"Body-X ($\text{m}$)")
    ax.set_ylabel(r"Body-Y ($\text{m}$)")
    # ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(True)

    safe_lbl = foot_label.replace(" ", "_")
    pdf_dir = os.path.join(output_dir, "foot_com_positions_body_frame", "xy_trajectory")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"foot_com_position_xy_trajectory_{safe_lbl.lower()}.pdf")
    fig.savefig(pdf_path, dpi=600)
    
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"foot_com_position_xy_trajectory_{safe_lbl.lower()}.pickle"), "wb") as f:
            pickle.dump(fig, f)

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
        subfolder: str
):
    # ---------- per-foot grid ----------
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=FIGSIZE)
    for i, ax in enumerate(axes.flat):
        ax.plot(sim_times, positions_array[:, i], label=f'{axis_label.lower()}_{foot_labels[i]}')
        first = True
        for start_timestep, end_timestep in compute_stance_segments(contact_state_array[:, i].astype(bool)):
            ax.axvspan(sim_times[start_timestep], sim_times[end_timestep - 1], color='gray', alpha=.3, label='in contact' if first else None)
            first = False
        draw_resets(ax, reset_times)
        ax.set_title(f'Foot {axis_label} ({frame_label.replace("_", " ")}) {foot_labels[i]} {"(CoM)" if frame_label == "body_frame" else "(toe tip)"}', fontsize=18)
        ax.set_ylabel(rf'{axis_label} ($\text{{m}}$)')
        ax.legend()
    axes[-1, 0].set_xlabel(r'Time ($\text{s}$)')

    subdir = os.path.join(output_dir, subfolder)
    os.makedirs(subdir, exist_ok=True)
    pdf = os.path.join(subdir, f'foot_pos_{axis_label.lower()}_grid.pdf')
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'foot_pos_{axis_label.lower()}_grid_{frame_label}.pickle'), 'wb') as f:
            pickle.dump(fig, f)

    # ---------- overview ----------
    fig_ov, ax = plt.subplots(figsize=(FIGSIZE[0], FIGSIZE[1]))
    for i, lbl in enumerate(foot_labels):
        ax.plot(sim_times, positions_array[:, i], label=lbl)
    draw_resets(ax, reset_times)
    ax.set_xlabel(r'Time ($\text{s}$)')
    ax.set_ylabel(rf'{axis_label} ($\text{{m}}$)')
    ax.set_title(f'Foot {axis_label} ({frame_label.replace("_", " ")}) {foot_labels[i]} {"(CoM)" if frame_label == "body_frame" else "(toe tip)"} overview', fontsize=18)
    ax.legend(ncol=2, loc='upper right')
    pdf = os.path.join(subdir, f'foot_pos_{axis_label.lower()}_overview.pdf')
    fig_ov.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'foot_pos_{axis_label.lower()}_overview_{frame_label}.pickle'), 'wb') as f:
            pickle.dump(fig_ov, f)

def _plot_foot_velocity_time_series(
        velocities_array: np.ndarray,
        axis_label: str,
        frame_label: str,
        sim_times: np.ndarray,
        foot_labels: list[str],
        contact_state_array: np.ndarray,
        reset_times: list[float],
        output_dir: str,
        pickle_dir: str,
        FIGSIZE: tuple[int, int],
        subfolder: str
):
    # ---------- per-foot grid ----------
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=FIGSIZE)
    for i, ax in enumerate(axes.flat):
        ax.plot(sim_times, velocities_array[:, i], label=f'vel_{axis_label.lower()}_{foot_labels[i]}')
        first = True
        for start_timestep, end_timestep in compute_stance_segments(contact_state_array[:, i].astype(bool)):
            ax.axvspan(sim_times[start_timestep], sim_times[end_timestep - 1], color='gray', alpha=.3, label='in contact' if first else None)
            first = False
        draw_resets(ax, reset_times)
        ax.set_title(f'Foot {axis_label} Velocity ({frame_label.replace("_", " ")}) {foot_labels[i]}', fontsize=18)
        ax.set_ylabel(rf'Velocity {axis_label} ($\text{{m}} \cdot \text{{s}}^{{-1}})$')
        ax.legend()
    axes[-1, 0].set_xlabel(r'Time ($\text{s}$)')

    subdir = os.path.join(output_dir, subfolder)
    os.makedirs(subdir, exist_ok=True)
    pdf = os.path.join(subdir, f'foot_vel_{axis_label.lower()}_grid.pdf')
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'foot_vel_{axis_label.lower()}_grid_{frame_label}.pickle'), 'wb') as f:
            pickle.dump(fig, f)
    plt.close(fig)

    # ---------- overview ----------
    fig_ov, ax = plt.subplots(figsize=(FIGSIZE[0], FIGSIZE[1]))
    for i, lbl in enumerate(foot_labels):
        ax.plot(sim_times, velocities_array[:, i], label=lbl)
    draw_resets(ax, reset_times)
    ax.set_xlabel(r'Time ($\text{s}$)')
    ax.set_ylabel(rf'Velocity {axis_label} ($\text{{m}} \cdot \text{{s}}^{{-1}})$')
    ax.set_title(f'Foot {axis_label} Velocity ({frame_label.replace("_", " ")}) overview', fontsize=18)
    ax.legend(ncol=2, loc='upper right')
    pdf = os.path.join(subdir, f'foot_vel_{axis_label.lower()}_overview.pdf')
    fig_ov.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'foot_vel_{axis_label.lower()}_overview_{frame_label}.pickle'), 'wb') as f:
            pickle.dump(fig_ov, f)
    plt.close(fig_ov)


def _plot_foot_velocity_time_series_single(
        velocities_array: np.ndarray,
        axis_label: str,
        frame_label: str,
        sim_times: np.ndarray,
        foot_label: str,
        foot_idx: int,
        contact_state_array: np.ndarray,
        reset_times: list[float],
        output_dir: str,
        pickle_dir: str,
        FIGSIZE: tuple[int, int],
        subfolder: str
):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, velocities_array[:, foot_idx], label=f'vel_{axis_label.lower()}_{foot_label}')
    first = True
    for start_timestep, end_timestep in compute_stance_segments(contact_state_array[:, foot_idx].astype(bool)):
        ax.axvspan(sim_times[start_timestep], sim_times[end_timestep - 1], color='gray', alpha=.3, label='in contact' if first else None)
        first = False
    draw_resets(ax, reset_times)
    ax.set_title(f'Foot {axis_label} Velocity ({frame_label.replace("_", " ")}) {foot_label}', fontsize=18)
    ax.set_ylabel(rf'Velocity {axis_label} ($\text{{m}} \cdot \text{{s}}^{{-1}})$')
    ax.set_xlabel(r'Time ($\text{s}$)')
    ax.legend()

    subdir = os.path.join(output_dir, subfolder)
    os.makedirs(subdir, exist_ok=True)
    safe_label = foot_label.replace(" ", "_").lower()
    pdf = os.path.join(subdir, f'foot_vel_{axis_label.lower()}_single_{safe_label}.pdf')
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'foot_vel_{axis_label.lower()}_single_{safe_label}_{frame_label}.pickle'), 'wb') as f:
            pickle.dump(fig, f)
    plt.close(fig)

def _plot_foot_velocity_magnitude_time_series(
        velocity_magnitudes_array: np.ndarray, # Shape (T, 4)
        frame_label: str,
        sim_times: np.ndarray,
        foot_labels: list[str],
        contact_state_array: np.ndarray,
        reset_times: list[float],
        output_dir: str,
        pickle_dir: str,
        FIGSIZE: tuple[int, int],
        subfolder: str
):
    # ---------- per-foot grid ----------
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=FIGSIZE)
    for i, ax in enumerate(axes.flat):
        ax.plot(sim_times, velocity_magnitudes_array[:, i], label=f'vel_mag_{foot_labels[i]}')
        first = True
        for start_timestep, end_timestep in compute_stance_segments(contact_state_array[:, i].astype(bool)):
            ax.axvspan(sim_times[start_timestep], sim_times[end_timestep - 1], color='gray', alpha=.3, label='in contact' if first else None)
            first = False
        draw_resets(ax, reset_times)
        ax.set_title(f'Foot Velocity Magnitude ({frame_label.replace("_", " ")}) {foot_labels[i]}', fontsize=18)
        ax.set_ylabel(rf'Velocity Magnitude ($\text{{m}} \cdot \text{{s}}^{{-1}})$')
        ax.legend()
    axes[-1, 0].set_xlabel(r'Time ($\text{s}$)')

    subdir = os.path.join(output_dir, subfolder)
    os.makedirs(subdir, exist_ok=True)
    pdf = os.path.join(subdir, f'foot_vel_magnitude_grid.pdf')
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'foot_vel_magnitude_grid_{frame_label}.pickle'), 'wb') as f:
            pickle.dump(fig, f)
    plt.close(fig)

    # ---------- overview ----------
    fig_ov, ax = plt.subplots(figsize=(FIGSIZE[0], FIGSIZE[1]))
    for i, lbl in enumerate(foot_labels):
        ax.plot(sim_times, velocity_magnitudes_array[:, i], label=lbl)
    draw_resets(ax, reset_times)
    ax.set_xlabel(r'Time ($\text{s}$)')
    ax.set_ylabel(rf'Velocity Magnitude ($\text{{m}} \cdot \text{{s}}^{{-1}})$')
    ax.set_title(f'Foot Velocity Magnitude ({frame_label.replace("_", " ")}) overview', fontsize=18)
    ax.legend(ncol=2, loc='upper right')
    pdf = os.path.join(subdir, f'foot_vel_magnitude_overview.pdf')
    fig_ov.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'foot_vel_magnitude_overview_{frame_label}.pickle'), 'wb') as f:
            pickle.dump(fig_ov, f)
    plt.close(fig_ov)

def _plot_foot_velocity_magnitude_time_series_single(
        velocity_magnitudes_array: np.ndarray, # Shape (T, 4)
        frame_label: str,
        sim_times: np.ndarray,
        foot_label: str,
        foot_idx: int,
        contact_state_array: np.ndarray,
        reset_times: list[float],
        output_dir: str,
        pickle_dir: str,
        FIGSIZE: tuple[int, int],
        subfolder: str
):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, velocity_magnitudes_array[:, foot_idx], label=f'vel_mag_{foot_label}')
    first = True
    for start_timestep, end_timestep in compute_stance_segments(contact_state_array[:, foot_idx].astype(bool)):
        ax.axvspan(sim_times[start_timestep], sim_times[end_timestep - 1], color='gray', alpha=.3, label='in contact' if first else None)
        first = False
    draw_resets(ax, reset_times)
    ax.set_title(f'Foot Velocity Magnitude ({frame_label.replace("_", " ")}) {foot_label}', fontsize=18)
    ax.set_ylabel(rf'Velocity Magnitude ($\text{{m}} \cdot \text{{s}}^{{-1}})$')
    ax.set_xlabel(r'Time ($\text{s}$)')
    ax.legend()

    subdir = os.path.join(output_dir, subfolder)
    os.makedirs(subdir, exist_ok=True)
    safe_label = foot_label.replace(" ", "_").lower()
    pdf = os.path.join(subdir, f'foot_vel_magnitude_single_{safe_label}.pdf')
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'foot_vel_magnitude_single_{safe_label}_{frame_label}.pickle'), 'wb') as f:
            pickle.dump(fig, f)
    plt.close(fig)

def _plot_hist_metric_grid(metric_dict: dict[str, list[float]], title: str, xlabel: str, foot_labels: list[str], output_dir: str, pickle_dir: str, subfolder: str, FIGSIZE: tuple[int, int]):
    """
    Generic 2x2 grid histogram for per-foot metrics in `metric_dict`.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=22)
    for i, lbl in enumerate(foot_labels):
        ax = axes.flat[i]
        data = metric_dict[lbl]
        if data:
            counts, edges = compute_trimmed_histogram_data(np.array(data))
            # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
            ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
        else:
            ax.text(.5, .5, 'no data', ha='center', va='center', transform=ax.transAxes, color='red')
        ax.set_title(lbl, fontsize=18)
        ax.set_xlabel(xlabel); ax.set_ylabel('Count')
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
            counts, edges = compute_trimmed_histogram_data(np.array(data))
            # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
            ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0, label=lbl)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right')
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
    fig.suptitle(title, fontsize=22)
    for i, lbl in enumerate(foot_labels):
        ax = axes.flat[i]
        data = metric_dict[lbl]
        if data:
            ax.boxplot(data, showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(lbl, fontsize=18); ax.set_xlabel(xlabel)
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
        ax.boxplot(data, positions=np.arange(1, len(lbls)+1), showmeans=True, showcaps=True, showbox=True, showfliers=False)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Foot'); ax.set_ylabel(xlabel)
    ax.set_xticks(np.arange(1, len(lbls)+1)); ax.set_xticklabels(lbls)
    pdf = os.path.join(output_dir, subfolder, f"box_{os.path.basename(subfolder)}_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"box_{os.path.basename(subfolder)}_overview.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_foot_contact_force_per_foot(sim_times, contact_forces_array, foot_labels, contact_state_array, reset_times, constraint_bounds, output_dir, pickle_dir):
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        ax.plot(sim_times, contact_forces_array[:, i], label=f'force_mag_{foot_labels[i]}')
        draw_limits(ax, "foot_contact_force", constraint_bounds)
        draw_resets(ax, reset_times)

        first = True
        for s, e in compute_stance_segments(in_contact=contact_state_array[:, i].astype(bool)):
            ax.axvspan(sim_times[s], sim_times[e-1], facecolor='gray', alpha=0.3, label='in contact' if first else None)
            first = False

        ax.set_title(f"Foot contact force magnitude {foot_labels[i]}", fontsize=20)
        ax.set_ylabel(r'Force ($\text{N}$)')
        ax.legend()
    pdf = os.path.join(output_dir, "foot_contact_forces",'foot_contact_force_grid.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'foot_contact_force_grid.pickle'), 'wb') as f:
            pickle.dump(fig, f)

def _plot_hist_contact_forces_grid(contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE):
    """
    Plots a 2×2 grid of histograms of contact forces for each foot, showing only forces > 0 N.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("Histogram of Foot Contact Forces", fontsize=22)
    
    # Loop over each foot label / column
    for i, label in enumerate(foot_labels):
        ax = axes.flat[i]
        forces = contact_forces_array[:, i]
        positive_forces = forces[forces > 0]
        
        if positive_forces.size == 0:
            # Warn if no positive forces were found
            ax.text(0.5, 0.5, "No forces > 0 N", ha="center", va="center", transform=ax.transAxes, fontsize=16, color="red")
        else:
            counts, edges = compute_trimmed_histogram_data(positive_forces)
            # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
            ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
        
        ax.set_title(label, fontsize=18)
        ax.set_xlabel(r"Force ($\text{N}$)")
        ax.set_ylabel("Count")
    
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
            counts, edges = compute_trimmed_histogram_data(positive_forces)
            # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
            ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0, label=label)
        else:
            # Warn in the plot that there's no data for this label
            ax.text(0.5, 0.5 - 0.05 * i, f"No >0 data for '{label}'", transform=ax.transAxes, fontsize=12, color='gray', ha='center')
    
    ax.set_title("Histogram of Foot Contact Forces", fontsize=20)
    ax.set_xlabel(r"Force ($\text{N}$)")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right', fontsize=12)
    
    pdf_path = os.path.join(output_dir, "foot_contact_forces", "hist_contact_forces_overview.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, dpi=600)
    
    pickle_path = os.path.join(pickle_dir, "hist_contact_forces_overview.pickle")
    if pickle_dir != "":
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)

def _plot_joint_metric(metric_name, data_arr, sim_times, joint_names, leg_row, leg_col, foot_from_joint, contact_state_array, reset_times, constraint_bounds, metric_to_constraint_term_mapping, metric_to_unit_mapping, output_dir, pickle_dir):
    fig, axes = plt.subplots(4, 3, sharex=True, figsize=(18, 12))
    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        if row is None or col is None:
            raise ValueError("Could not determine joint row/col for plotting")
        ax = axes[row, col]
        ax.plot(sim_times, data_arr[:, j])
        if metric_name == 'position':
            draw_limits(ax, jn, constraint_bounds)
        else:
            draw_limits(ax, metric_to_constraint_term_mapping[metric_name], constraint_bounds)
        draw_resets(ax, reset_times)

        fid = foot_from_joint[j]
        if fid is not None:
            for stance_start, stance_end in compute_stance_segments(contact_state_array[:, fid].astype(bool)):
                ax.axvspan(sim_times[stance_start], sim_times[stance_end-1], facecolor='gray', alpha=0.5)
        ax.set_title(f"Joint {metric_name.replace('_', ' ')} for {jn}", fontsize=20)
        ax.set_ylabel(f"Joint {metric_name.replace('_', ' ').capitalize()} ({metric_to_unit_mapping[metric_name]})")

    axes[-1, 0].set_xlabel(r'Time ($\text{s}$)')
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f'joint_{metric_name}_grid.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f'joint_{metric_name}_grid.pickle'), 'wb') as f:
            pickle.dump(fig, f)

    # overview
    fig_ov, ax = plt.subplots(figsize=(12, 6))
    for j in range(data_arr.shape[1]):
        ax.plot(sim_times, data_arr[:, j], label=joint_names[j], linestyle=get_leg_linestyle(joint_names[j]))
    if metric_name == 'position':
        for jn in joint_names:
            draw_limits(ax, jn, constraint_bounds)
    else:
        draw_limits(ax, metric_to_constraint_term_mapping[metric_name], constraint_bounds)
        draw_resets(ax, reset_times)
        ax.set_xlabel(r'Time ($\text{s}$)')
        ax.set_title(f"Joint {metric_name.replace('_', ' ')} overview", fontsize=20)
        ax.set_ylabel(f"Joint {metric_name.replace('_', ' ').capitalize()} ({metric_to_unit_mapping[metric_name]})")
        ax.legend(loc='upper right', ncol=2)
        pdf = os.path.join(output_dir, "joint_metrics", metric_name, f'joint_{metric_name}_overview.pdf')
        fig_ov.savefig(pdf, dpi=600)
        if pickle_dir != "":
            with open(os.path.join(pickle_dir, f'joint_{metric_name}_overview.pickle'), 'wb') as f:
                pickle.dump(fig_ov, f)

def _plot_hist_joint_grid(metric_name, data_arr, joint_names, leg_row, leg_col, metric_to_unit_mapping, output_dir, pickle_dir):
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle(f"Histogram of Joint {metric_name.replace('_', ' ').title()}", fontsize=22)
    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        ax = axes[row, col]

        counts, edges = compute_trimmed_histogram_data(data_arr[:, j])
        # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
        ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
        ax.set_title(jn, fontsize=16)
        ax.set_xlabel(f"{metric_name.capitalize().replace('_', ' ')} ({metric_to_unit_mapping[metric_name]})")
        ax.set_ylabel("Count")
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"hist_joint_{metric_name}_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"hist_joint_{metric_name}_grid.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_hist_joint_metric_overview(metric_name, data_arr, joint_names, metric_to_unit_mapping, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for j, jn in enumerate(joint_names):
        counts, edges = compute_trimmed_histogram_data(data_arr[:, j])
        # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
        ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0, label=jn)
    ax.set_title(f"Histogram of joint {metric_name.replace('_',' ')}", fontsize=20)
    ax.set_xlabel(f"Joint {metric_name.replace('_',' ').capitalize()} ({metric_to_unit_mapping[metric_name]})")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right', fontsize=12)
    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"hist_joint_{metric_name}_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"hist_joint_{metric_name}_overview.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_hist_air_time_per_foot_grid(air_segments_per_foot, foot_labels, output_dir, pickle_dir, FIGSIZE):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("Histogram of Air-Time Durations per Foot", fontsize=22)
    for i, label in enumerate(foot_labels):
        durations = air_segments_per_foot[label]
        ax = axes.flat[i]
        if durations:
            counts, edges = compute_trimmed_histogram_data(np.array(durations))
            # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
            ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
        ax.set_title(label, fontsize=18)
        ax.set_xlabel(r"Air Time ($\text{s}$)")
        ax.set_ylabel("Count")
    pdf = os.path.join(output_dir, "aggregates", "air_time", "hist_air_time_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "hist_air_time_grid.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_hist_air_time_per_foot_single(label, durations, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if durations:
        counts, edges = compute_trimmed_histogram_data(np.array(durations))
        # ax.stairs is faster to draw but we have to manually make it look like default ax.hist
        ax.stairs(counts, edges, fill=True, linewidth=plt.rcParams["patch.linewidth"], alpha=0.7, edgecolor=None, baseline=0)
    ax.set_title(f"Histogram of Air-Time Durations ({label})", fontsize=20)
    ax.set_xlabel(r"Air Time ($\text{s}$)")
    ax.set_ylabel("Frequency")
    safe_label = label.replace(' ', '_')
    pdf = os.path.join(output_dir, "aggregates", "air_time", f"hist_air_time_{safe_label}.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"hist_air_time_{safe_label}.pickle"), 'wb') as f:
            pickle.dump(fig, f)

def _plot_combined_energy(sim_times, combined_energy, reset_times, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, combined_energy, label='total_energy')
    draw_resets(ax, reset_times)
    ax.set_xlabel(r'Time ($\text{s}$)')
    ax.set_ylabel(r'Energy ($\text{J}$)')
    ax.set_title('Total Cumulative Joint Energy', fontsize=20)
    ax.legend()
    pdf = os.path.join(output_dir, "aggregates", 'combined_energy_overview.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'combined_energy_overview.pickle'), 'wb') as f:
            pickle.dump(fig, f)

def _plot_reward_time_series(sim_times, reward_array, reset_times, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, reward_array, label='Reward')
    draw_resets(ax, reset_times)
    ax.set_xlabel(r'Time ($\text{s}$)')
    ax.set_ylabel('Reward (-)')
    ax.set_title('Reward at each time step', fontsize=20)
    ax.legend()
    pdf = os.path.join(output_dir, "aggregates", 'reward_time_series.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'reward_time_series.pickle'), 'wb') as f:
            pickle.dump(fig, f)

def _plot_cumulative_reward(sim_times, reward_array, reset_times, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, np.cumsum(reward_array), label='Cumulative Reward')
    draw_resets(ax, reset_times)
    ax.set_xlabel(r'Time ($\text{s}$)')
    ax.set_ylabel('Cumulative reward (-)')
    ax.set_title('Cumulative reward', fontsize=20)
    ax.legend()
    pdf = os.path.join(output_dir, "aggregates", 'cumulative_reward.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'cumulative_reward.pickle'), 'wb') as f:
            pickle.dump(fig, f)

def _plot_cost_of_transport(sim_times, cost_of_transport_time_series, reset_times, output_dir, pickle_dir, FIGSIZE):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sim_times, cost_of_transport_time_series, label='cost_of_transport')
    draw_resets(ax, reset_times)
    # Plot running average
    # window_sizes = [25, 50, 300]
    window_sizes = [100]
    for window_size in window_sizes:
        window = np.ones(window_size) / window_size
        running_average = np.convolve(cost_of_transport_time_series, window, mode='valid')
        # Align the running average times. For 'valid', the i-th averaged point corresponds to sim_times[i + (window_size-1)], i.e. the right edge of the window.
        average_times = sim_times[(window_size - 1):]
        ax.plot(average_times, running_average, label=f'{window_size}-sample running avg', linestyle="dashed")

    ax.set_xlabel(r'Time ($\text{s}$)')
    ax.set_ylabel('Cost of Transport (-)')
    ax.set_title('Instantaneous Cost of Transport', fontsize=20)
    ax.set_ylim(0, 6)
    ax.legend()
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
    ax.set_xlabel('Cost of Transport (-)')
    ax.set_xlim(0, 6)
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Cost of Transport', fontsize=20)
    pdf = os.path.join(output_dir, "aggregates", "cost_of_transport", 'hist_cot_overview.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'hist_cot_overview.pickle'), 'wb') as f:
            pickle.dump(fig, f)

def _plot_combined_base_position(sim_times, base_position_array, reset_times, output_dir, pickle_dir, FIGSIZE):
    fig_bp, axes_bp = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, axis_label in enumerate(['X', 'Y', 'Z']):
        axes_bp[i].plot(sim_times, base_position_array[:, i], label=f'position_{axis_label}')
        draw_resets(axes_bp[i], reset_times)
        axes_bp[i].set_title(f'World Position {axis_label}')
        axes_bp[i].set_ylabel(r'Position ($\text{m}$)')
        axes_bp[i].legend()
    axes_bp[-1].set_xlabel(r'Time ($\text{s}$)')
    pdf = os.path.join(output_dir, "base_kinematics", 'base_position_subplots_world.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_bp.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'base_position_subplots_world.pickle'), 'wb') as f:
            pickle.dump(fig_bp, f)

def _plot_combined_orientation(sim_times, base_orientation_array, reset_times, output_dir, pickle_dir, FIGSIZE):
    fig_bo, axes_bo = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, orient_label in enumerate(['Yaw', 'Pitch', 'Roll']):
        axes_bo[i].plot(sim_times, base_orientation_array[:, i], label=orient_label)
        draw_resets(axes_bo[i], reset_times)
        axes_bo[i].set_ylabel(rf'{orient_label} ($\text{{rad}}$)')
        axes_bo[i].set_title(f'World Orientation {orient_label}')
        axes_bo[i].legend()
    axes_bo[-1].set_xlabel(r'Time ($\text{s}$)')
    pdf = os.path.join(output_dir, "base_kinematics", 'base_orientation_subplots.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_bo.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'base_orientation_subplots.pickle'), 'wb') as f:
            pickle.dump(fig_bo, f)

def _plot_combined_base_velocity(sim_times, base_linear_velocity_array, commanded_velocity_array, reset_times, output_dir, pickle_dir, FIGSIZE):
    fig_blv, axes_blv = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, vel_label in enumerate(['Velocity X', 'Velocity Y', 'Velocity Z']):
        axes_blv[i].plot(sim_times, base_linear_velocity_array[:, i], label=vel_label)
        if i != 2:
            axes_blv[i].plot(sim_times, commanded_velocity_array[:, i], linestyle='--', label=f'cmd_{vel_label}')
        draw_resets(axes_blv[i], reset_times)
        axes_blv[i].set_ylabel(rf"$\text{{{vel_label.replace(' ', '_')}}} (\text{{m}} \cdot \text{{s}}^{{-1}})$")
        axes_blv[i].set_title(f'Base Linear {vel_label} (Body Frame)')
        axes_blv[i].legend()
    axes_blv[-1].set_xlabel(r'Time ($\text{s}$)')
    pdf = os.path.join(output_dir, "base_kinematics", 'base_linear_velocity_subplots.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_blv.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'base_linear_velocity_subplots.pickle'), 'wb') as f:
            pickle.dump(fig_blv, f)

def _plot_combined_base_angular_velocities(sim_times, base_angular_velocity_array, commanded_velocity_array, reset_times, output_dir, pickle_dir, FIGSIZE):
    fig_bav, axes_bav = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    for i, vel_label in enumerate([r'$\Omega_X$', r'$\Omega_Y$', r'$\Omega_Z$']):
        axes_bav[i].plot(sim_times, base_angular_velocity_array[:, i], label=vel_label)
        if i == 2:
            axes_bav[i].plot(sim_times, commanded_velocity_array[:, i], linestyle='--', label=f'cmd_{vel_label}')
        draw_resets(axes_bav[i], reset_times)
        axes_bav[i].set_ylabel(rf"{vel_label} ($\text{{rad}} \cdot \text{{s}}^{{-1}})$")
        axes_bav[i].set_title(f'Base {vel_label} (Body Frame)')
        axes_bav[i].legend()
    axes_bav[-1].set_xlabel(r'Time ($\text{s}$)')
    pdf = os.path.join(output_dir, "base_kinematics", 'base_angular_velocity_subplots.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_bav.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'base_angular_velocity_subplots.pickle'), 'wb') as f:
            pickle.dump(fig_bav, f)

def _plot_total_base_overview(sim_times, base_position_array, base_orientation_array, base_linear_velocity_array, base_angular_velocity_array, reset_times, output_dir, pickle_dir, FIGSIZE):
    fig_overview, overview_axes = plt.subplots(2, 2, figsize=(20, 16))
    arrays = [base_position_array, base_orientation_array, base_linear_velocity_array, base_angular_velocity_array]
    titles = ['Base World Position', 'Base Orientation', 'Base Linear Velocity', 'Base Angular Velocity']
    labels = [['X', 'Y', 'Z'], ['Yaw', 'Pitch', 'Roll'], ['VX', 'VY', 'VZ'], [r'$W_X$', r'$W_Y$', r'$W_Z$']]
    for ax, arr, title, axis_labels in zip(overview_axes.flatten(), arrays, titles, labels):
        for i, lbl in enumerate(axis_labels):
            ax.plot(sim_times, arr[:, i], label=lbl)
            draw_resets(ax, reset_times)
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(r'Time ($\text{s}$)')
        ax.legend()
    pdf = os.path.join(output_dir, "base_kinematics", 'base_overview_world.pdf')
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig_overview.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, 'base_overview_world.pickle'), 'wb') as f:
            pickle.dump(fig_overview, f)

def _plot_command_abs_error_base_kinematics(
    sim_times,
    base_linear_velocity_array,
    base_angular_velocity_array,
    commanded_velocity_array,
    reset_times,
    output_dir,
    pickle_dir,
    FIGSIZE
):
    """
    Plots the absolute error between commanded and actual base kinematic components (Vx, Vy, Omega_Z) over time.
    Generates a grid plot (3x1) and an overview plot.
    """
    # Calculate absolute errors
    # commanded_velocity_array[:, 0] is cmd_vx
    # base_linear_velocity_array[:, 0] is actual_vx
    abs_err_vx = np.abs(commanded_velocity_array[:, 0] - base_linear_velocity_array[:, 0])

    # commanded_velocity_array[:, 1] is cmd_vy
    # base_linear_velocity_array[:, 1] is actual_vy
    abs_err_vy = np.abs(commanded_velocity_array[:, 1] - base_linear_velocity_array[:, 1])

    # commanded_velocity_array[:, 2] is cmd_omega_z (yaw rate)
    # base_angular_velocity_array[:, 2] is actual_omega_z (yaw rate)
    abs_err_omega_z = np.abs(commanded_velocity_array[:, 2] - base_angular_velocity_array[:, 2])

    error_arrays = [abs_err_vx, abs_err_vy, abs_err_omega_z]
    error_labels_full = [ # Full labels for titles etc.
        'Abs. Err. Lin. Vel. X',
        'Abs. Err. Lin. Vel. Y',
        'Abs. Err. Ang. Vel. Z (Yaw)'
    ]
    error_labels_short = [ # Short labels for y-axis and individual titles
        'Lin. Vel. X',
        'Lin. Vel. Y',
        'Ang. Vel. Z (Yaw)'
    ]
    error_units = [
        r'$\text{m/s}$',
        r'$\text{m/s}$',
        r'$\text{rad/s}$'
    ]

    # Grid Plot (3 subplots, 1 for each error component)
    fig_grid, axes_grid = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)
    # fig_grid.suptitle('Commanded vs Actual Base Kinematics Absolute Error', fontsize=18) # Optional overall title

    for i, ax in enumerate(axes_grid.flat):
        ax.plot(sim_times, error_arrays[i], label=f'Abs. Error {error_labels_short[i]}')
        draw_resets(ax, reset_times)
        ax.set_ylabel(f'Abs. Error ({error_units[i]})')
        ax.set_title(f'Absolute Error: Cmd vs Actual Base {error_labels_short[i]}')
        ax.legend(loc='upper right')
    axes_grid[-1].set_xlabel(r'Time ($\text{s}$)')
    fig_grid.tight_layout() # Adjust layout

    plot_dir = os.path.join(output_dir, "base_kinematics")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path_grid = os.path.join(plot_dir, 'base_command_abs_error_subplots.pdf')
    fig_grid.savefig(plot_path_grid, dpi=600)

    if pickle_dir != "":
        pickle_plot_dir = os.path.join(pickle_dir, "base_kinematics")
        os.makedirs(pickle_plot_dir, exist_ok=True)
        with open(os.path.join(pickle_plot_dir, 'base_command_abs_error_subplots.pickle'), 'wb') as f:
            pickle.dump(fig_grid, f)
    plt.close(fig_grid)

    # Overview Plot
    fig_overview, ax_overview = plt.subplots(figsize=FIGSIZE)
    ax_overview.set_title('Commanded vs Actual Base Kinematics Absolute Error (Overview)', fontsize=18)
    
    for i in range(len(error_arrays)):
        ax_overview.plot(sim_times, error_arrays[i], label=f'{error_labels_full[i]} ({error_units[i]})')

    draw_resets(ax_overview, reset_times)
    ax_overview.set_xlabel(r'Time ($\text{s}$)')
    ax_overview.set_ylabel(r'Absolute Error')
    ax_overview.legend(loc='upper right', ncol=1)
    fig_overview.tight_layout()

    plot_path_overview = os.path.join(plot_dir, 'base_command_abs_error_overview.pdf')
    fig_overview.savefig(plot_path_overview, dpi=600)
    if pickle_dir != "":
        # pickle_plot_dir already created
        with open(os.path.join(pickle_plot_dir, 'base_command_abs_error_overview.pickle'), 'wb') as f:
            pickle.dump(fig_overview, f)
    plt.close(fig_overview)


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
    fig.suptitle(f"box Plot of Joint {metric_name.replace('_', ' ').title()}", fontsize=22)

    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        ax = axes[row, col]
        ax.boxplot(x=data_arr[:, j], showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(jn, fontsize=16)
        ax.set_xlabel(f"{metric_name.capitalize().replace('_', ' ')} ({metric_to_unit_mapping[metric_name]})", fontsize=12)

    pdf = os.path.join(output_dir, "joint_metrics", metric_name, f"box_joint_{metric_name}_grid.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, f"box_joint_{metric_name}_grid.pickle"), "wb") as f:
            pickle.dump(fig, f)


def _plot_box_joint_metric_overview(metric_name, data_arr, joint_names, metric_to_unit_mapping, output_dir, pickle_dir, FIGSIZE):
    """
    Overview box plot (all joints on one axis) for the given metric.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.boxplot(x=[data_arr[:, j] for j in range(data_arr.shape[1])], positions=np.arange(1, len(joint_names) + 1), showmeans=True, showcaps=True, showbox=True, showfliers=False)
    ax.set_title(f"Box Plot of Joint {metric_name.replace('_', ' ')}", fontsize=20)
    ax.set_xlabel("Joint")
    ax.set_ylabel(f"{metric_name.capitalize().replace('_', ' ')} ({metric_to_unit_mapping[metric_name]})")
    ax.set_xticks(np.arange(1, len(joint_names) + 1))
    ax.set_xticklabels(joint_names, rotation=45, ha="right")

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
    fig.suptitle("box Plot of Foot Contact Forces", fontsize=22)

    for i, label in enumerate(foot_labels):
        ax = axes.flat[i]
        forces = contact_forces_array[:, i]
        positive = forces[forces > 0]
        if positive.size == 0:
            ax.text(0.5, 0.5, "No forces > 0 N", ha="center", va="center", transform=ax.transAxes, fontsize=16, color="red")
        else:
            ax.boxplot(positive, showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(label, fontsize=18)
        ax.set_xlabel(r"Force ($\text{N}$)")

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
        ax.boxplot(data, positions=np.arange(1, len(data) + 1), showmeans=True, showcaps=True, showbox=True, showfliers=False)
    ax.set_title("Box Plot of Foot Contact Forces", fontsize=20)
    ax.set_xlabel("Foot")
    ax.set_ylabel(r"Force ($\text{N}$)")
    ax.set_xticks(np.arange(1, len(used_labels) + 1))
    ax.set_xticklabels(used_labels)

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
    fig.suptitle("box Plot of Air-Time Durations per Foot", fontsize=22)

    for i, label in enumerate(foot_labels):
        durations = air_segments_per_foot[label]
        ax = axes.flat[i]
        if durations:
            ax.boxplot(durations, showmeans=True, showcaps=True, showbox=True, showfliers=False)
        ax.set_title(label, fontsize=18)
        ax.set_xlabel(r"Air Time ($\text{s}$)")

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
    else:
        ax.text(0.5, 0.5, "No air-time segments", ha="center", va="center", transform=ax.transAxes, fontsize=16, color="red")

    ax.set_title(f"Box Plot of Air-Time Durations ({label})", fontsize=20)
    ax.set_xlabel(r"Air Time ($\text{s}$)")

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
    ax.set_xlabel("Cost of Transport (-)")
    ax.set_title("Box Plot of Cost of Transport", fontsize=20)

    pdf = os.path.join(output_dir, "aggregates", "cost_of_transport", "box_cot_overview.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf, dpi=600)
    if pickle_dir != "":
        with open(os.path.join(pickle_dir, "box_cot_overview.pickle"), "wb") as f:
            pickle.dump(fig, f)


def _plot_foot_velocity_vs_height_grid(
    foot_velocities_world_frame: np.ndarray,
    foot_positions_contact_frame: np.ndarray,
    foot_labels: list[str],
    output_dir: str,
    pickle_dir: str,
    height_threshold: float,
    FIGSIZE: tuple[int, int]
):
    """
    2x2 grid of scatter plots showing foot world velocity magnitude vs. foot height in contact frame.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(f"Foot World Velocity vs. Height (up to {height_threshold:.2f}m)", fontsize=22)

    for i, (ax, lbl) in enumerate(zip(axes.flat, foot_labels)):
        heights = foot_positions_contact_frame[:, i, 2]
        velocities = np.linalg.norm(foot_velocities_world_frame[:, i, :], axis=1)
        
        mask = heights <= height_threshold
        
        ax.scatter(heights[mask], velocities[mask], alpha=0.5, s=5)
        ax.set_title(lbl, fontsize=18)
        ax.set_xlabel(r"Foot Height (contact frame) ($\text{m}$)")
        ax.set_ylabel(r"Foot Velocity (world frame) ($\text{m} \cdot \text{s}^{-1}$)")
        ax.grid(True)

    subfolder = "foot_velocity_vs_height"
    pdf_path = os.path.join(output_dir, subfolder, "foot_velocity_vs_height_grid.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, dpi=600)
    
    if pickle_dir != "":
        pickle_path = os.path.join(pickle_dir, "foot_velocity_vs_height_grid.pickle")
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)
    
    plt.close(fig)


def _plot_foot_velocity_vs_height_overview(
    foot_velocities_world_frame: np.ndarray,
    foot_positions_contact_frame: np.ndarray,
    foot_labels: list[str],
    output_dir: str,
    pickle_dir: str,
    height_threshold: float,
    FIGSIZE: tuple[int, int]
):
    """
    Overlayed scatter plot of foot world velocity magnitude vs. foot height for all feet.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.suptitle(f"Foot World Velocity vs. Height (up to {height_threshold:.2f}m) Overview", fontsize=22)

    for i, lbl in enumerate(foot_labels):
        heights = foot_positions_contact_frame[:, i, 2]
        velocities = np.linalg.norm(foot_velocities_world_frame[:, i, :], axis=1)
        
        mask = heights <= height_threshold
        
        ax.scatter(heights[mask], velocities[mask], alpha=0.5, s=5, label=lbl)

    ax.set_xlabel(r"Foot Height (contact frame) ($\text{m}$)")
    ax.set_ylabel(r"Foot Velocity (world frame) ($\text{m} \cdot \text{s}^{-1}$)")
    ax.grid(True)
    ax.legend()

    subfolder = "foot_velocity_vs_height"
    pdf_path = os.path.join(output_dir, subfolder, "foot_velocity_vs_height_overview.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, dpi=600)
    
    if pickle_dir != "":
        pickle_path = os.path.join(pickle_dir, "foot_velocity_vs_height_overview.pickle")
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)
    
    plt.close(fig)


def _plot_foot_velocity_vs_height_single(
    foot_velocities_world_frame: np.ndarray,
    foot_positions_contact_frame: np.ndarray,
    foot_idx: int,
    foot_label: str,
    output_dir: str,
    pickle_dir: str,
    height_threshold: float,
    FIGSIZE: tuple[int, int]
):
    """
    Scatter plot of foot world velocity magnitude vs. foot height for a single foot.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    heights = foot_positions_contact_frame[:, foot_idx, 2]
    velocities = np.linalg.norm(foot_velocities_world_frame[:, foot_idx, :], axis=1)
    
    mask = heights <= height_threshold
    
    ax.scatter(heights[mask], velocities[mask], alpha=0.5, s=5)
    ax.set_title(f"Foot World Velocity vs. Height ({foot_label})", fontsize=20)
    ax.set_xlabel(r"Foot Height (contact frame) ($\text{m}$)")
    ax.set_ylabel(r"Foot Velocity (world frame) ($\text{m} \cdot \text{s}^{-1}$)")
    ax.grid(True)

    subfolder = "foot_velocity_vs_height"
    safe_label = foot_label.replace(" ", "_")
    pdf_path = os.path.join(output_dir, subfolder, f"foot_velocity_vs_height_{safe_label}.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, dpi=600)
    
    if pickle_dir != "":
        pickle_path = os.path.join(pickle_dir, f"foot_velocity_vs_height_{safe_label}.pickle")
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)
    
    plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# Joint Phase Plots
# ----------------------------------------------------------------------------------------------------------------------

def _plot_joint_phase_overview(
    data_x: np.ndarray,
    data_y: np.ndarray,
    foot_labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (12, 12),
    gridsize: int = 100
):
    """Generates an overview hexbin plot for joint phase data."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(title, fontsize=28)

    x_flat = data_x.flatten()
    y_flat = data_y.flatten()

    if len(x_flat) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        hb = ax.hexbin(x_flat, y_flat, gridsize=gridsize, cmap='viridis', mincnt=1)
        counts = hb.get_array()
        
        cbar_label = "Occupancy (samples)"
        if counts.size > 0 and counts.max() > 10 and (counts.max() / (counts.min() + 1e-9) > 20):
            hb.set_norm(colors.LogNorm(vmin=max(1, counts.min()), vmax=counts.max()))
            cbar_label = r"$\log_{10}(\text{Occupancy})$"

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label(cbar_label, size=22)
        cb.ax.tick_params(labelsize=20)

    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "overview.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')
    
    if pickle_dir:
        os.makedirs(pickle_dir, exist_ok=True)
        with open(os.path.join(pickle_dir, "overview.pickle"), "wb") as f:
            pickle.dump(fig, f)
    plt.close(fig)


def _plot_joint_phase_grid(
    data_x: np.ndarray,
    data_y: np.ndarray,
    foot_labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (20, 20),
    gridsize: int = 100
):
    """Generates a 2x2 grid of hexbin plots for joint phase data."""
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=30)

    for i, (ax, lbl) in enumerate(zip(axes.flat, foot_labels)):
        x = data_x[:, i]
        y = data_y[:, i]

        ax.set_title(lbl, fontsize=26)
        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap='viridis', mincnt=1)
        counts = hb.get_array()
        
        cbar_label = "Occupancy (samples)"
        if counts.size > 0 and counts.max() > 10 and (counts.max() / (counts.min() + 1e-9) > 20):
            hb.set_norm(colors.LogNorm(vmin=max(1, counts.min()), vmax=counts.max()))
            cbar_label = r"$\log_{10}(\text{Occupancy})$"

        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label(cbar_label, size=22)
        cb.ax.tick_params(labelsize=20)

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "grid.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')

    if pickle_dir:
        os.makedirs(pickle_dir, exist_ok=True)
        with open(os.path.join(pickle_dir, "grid.pickle"), "wb") as f:
            pickle.dump(fig, f)
    plt.close(fig)


def _plot_joint_phase_single(
    data_x: np.ndarray,
    data_y: np.ndarray,
    foot_label: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (12, 12),
    gridsize: int = 100
):
    """Generates a single hexbin plot for joint phase data for one foot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(f"{title} - {foot_label}", fontsize=28)

    if len(data_x) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        hb = ax.hexbin(data_x, data_y, gridsize=gridsize, cmap='viridis', mincnt=1)
        counts = hb.get_array()
        
        cbar_label = "Occupancy (samples)"
        if counts.size > 0 and counts.max() > 10 and (counts.max() / (counts.min() + 1e-9) > 20):
            hb.set_norm(colors.LogNorm(vmin=max(1, counts.min()), vmax=counts.max()))
            cbar_label = r"$\log_{10}(\text{Occupancy})$"

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label(cbar_label, size=22)
        cb.ax.tick_params(labelsize=20)

    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)

    safe_lbl = foot_label.replace(" ", "_").lower()
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"single_{safe_lbl}.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')
    
    if pickle_dir:
        os.makedirs(pickle_dir, exist_ok=True)
        with open(os.path.join(pickle_dir, f"single_{safe_lbl}.pickle"), "wb") as f:
            pickle.dump(fig, f)
    plt.close(fig)

def _prepare_stance_only_velocity_data(
    foot_velocities_all_frames: np.ndarray, # (T, 4, 3) for components, or (T, 4) for magnitude
    contact_state_array: np.ndarray, # (T, 4)
    foot_labels: list[str],
    component_idx: int | None = None # 0 for X, 1 for Y, 2 for Z, None for magnitude
) -> dict[str, list[float]]:
    stance_data_dict = {lbl: [] for lbl in foot_labels}
    num_timesteps_velocities = foot_velocities_all_frames.shape[0]
    num_timesteps_contact = contact_state_array.shape[0]

    # Ensure we don't go out of bounds if arrays have different T (e.g. due to slicing)
    # This can happen if sim_data.npz is from a very short run.
    # We assume contact_state_array dictates the valid range of timesteps.
    
    if num_timesteps_velocities == 0 or num_timesteps_contact == 0:
        return stance_data_dict # Not enough data

    for foot_i, lbl in enumerate(foot_labels):
        # Get all timesteps where this foot is in contact based on contact_state_array
        stance_indices_for_foot_contact_time = np.where(contact_state_array[:, foot_i])[0]
        
        if len(stance_indices_for_foot_contact_time) == 0:
            continue

        # Filter these indices to be valid for the foot_velocities_all_frames array
        valid_stance_indices_for_velocities = stance_indices_for_foot_contact_time[stance_indices_for_foot_contact_time < num_timesteps_velocities]
        
        if len(valid_stance_indices_for_velocities) == 0:
            continue
            
        if component_idx is not None: # Components
            # foot_velocities_all_frames has shape (T_vel, 4, 3)
            velocities_this_foot_this_component_stance = foot_velocities_all_frames[valid_stance_indices_for_velocities, foot_i, component_idx]
        else: # Magnitude
            # foot_velocities_all_frames has shape (T_vel, 4)
            velocities_this_foot_this_component_stance = foot_velocities_all_frames[valid_stance_indices_for_velocities, foot_i]
        
        if velocities_this_foot_this_component_stance.size > 0:
            stance_data_dict[lbl].extend(velocities_this_foot_this_component_stance.tolist())
    return stance_data_dict


def _plot_foot_velocity_time_series_stance_focused(
    sim_times: np.ndarray,
    velocity_data: np.ndarray,  # Shape (T, 4) - can be one component or magnitude
    component_label: str,  # e.g., "VX", "VY", "VZ", "Magnitude"
    frame_label: str, # "world_frame"
    foot_labels: list[str],
    contact_state_array: np.ndarray, # Shape (T, 4)
    reset_times: list[float],
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int],
    subfolder_prefix: str,
    padding_steps: int = 1
):
    plot_subdir = os.path.join(output_dir, subfolder_prefix, component_label.lower().replace(" ", "_").replace("/", "_")) # Ensure component_label is filename-safe
    os.makedirs(plot_subdir, exist_ok=True)
    
    pickle_subdir = ""
    if pickle_dir: # Ensure pickle_dir is not an empty string
        pickle_subdir = os.path.join(pickle_dir, subfolder_prefix, component_label.lower().replace(" ", "_").replace("/", "_"))
        os.makedirs(pickle_subdir, exist_ok=True)

    # ---------- per-foot grid (stance-focused line plot) ----------
    fig_grid, axes_grid = plt.subplots(2, 2, sharex=True, figsize=FIGSIZE)
    # fig_grid.suptitle(f'Stance-Focused Foot {component_label} Velocity ({frame_label.replace("_", " ")})', fontsize=18)
    fig_grid.suptitle(f'Stance Foot {component_label} Vel. ({frame_label.replace("_", " ")}, {padding_steps} steps padding)', fontsize=18)


    for i, ax in enumerate(axes_grid.flat):
        foot_idx = i
        # Ensure contact_state_array has data for this foot_idx
        if foot_idx >= contact_state_array.shape[1]:
            ax.text(0.5, 0.5, "No data for this foot index", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f'{foot_labels[foot_idx]} (Error)', fontsize=16)
            continue

        stance_segments = compute_stance_segments(contact_state_array[:, foot_idx].astype(bool))
        
        first_segment_in_plot = True
        first_stance_span_for_legend = True
        plotted_anything = False

        if not stance_segments: # No stance phases for this foot
            ax.text(0.5, 0.5, "No stance data", ha="center", va="center", transform=ax.transAxes)
        
        for s_start, s_end in stance_segments:
            s_pad_start = max(0, s_start - padding_steps)
            # s_pad_end needs to be relative to sim_times length, and also velocity_data length
            s_pad_end = min(min(len(sim_times), velocity_data.shape[0]), s_end + padding_steps)


            if s_pad_start >= s_pad_end: # Skip if padded segment is empty or invalid
                continue

            times_to_plot = sim_times[s_pad_start:s_pad_end]
            # Ensure velocity_data has data for this foot_idx
            if foot_idx >= velocity_data.shape[1]:
                continue # Should not happen if contact_state_array check passed, but defensive
            
            data_to_plot = velocity_data[s_pad_start:s_pad_end, foot_idx]
            
            if times_to_plot.size == 0 or data_to_plot.size == 0: # Skip if no data after padding
                continue

            ax.plot(times_to_plot, data_to_plot, label=f'{component_label}_{foot_labels[foot_idx]}' if first_segment_in_plot else None)
            first_segment_in_plot = False
            plotted_anything = True

            # Highlight the actual stance phase within the padded segment
            # Ensure s_start and s_end-1 are valid indices for sim_times
            if s_start < len(sim_times) and s_end > 0 and s_start < s_end:
                actual_s_end_time_idx = min(s_end - 1, len(sim_times) - 1) # s_end is exclusive, so use s_end-1
                if actual_s_end_time_idx >= s_start : # Ensure valid span
                     ax.axvspan(sim_times[s_start], sim_times[actual_s_end_time_idx], color='gray', alpha=0.3, label='stance phase' if first_stance_span_for_legend else None)
                     if first_stance_span_for_legend:
                         first_stance_span_for_legend = False
        
        draw_resets(ax, reset_times)
        ax.set_title(f'{foot_labels[foot_idx]}', fontsize=16)
        ax.set_ylabel(rf'Velocity {component_label} ($\text{{m}} \cdot \text{{s}}^{{-1}})$')
        
        if plotted_anything or not first_stance_span_for_legend: # Add legend if plots were made or if stance highlight was added
             ax.legend(loc='best')


    axes_grid[-1, 0].set_xlabel(r'Time ($\text{s}$)')
    axes_grid[-1, 1].set_xlabel(r'Time ($\text{s}$)')
    
    pdf_grid_path = os.path.join(plot_subdir, f'line_grid_stance_focused.pdf')
    fig_grid.savefig(pdf_grid_path, dpi=600)
    if pickle_subdir: # Check again if pickle_subdir is valid
        with open(os.path.join(pickle_subdir, f'line_grid_stance_focused.pickle'), 'wb') as f:
            pickle.dump(fig_grid, f)
    plt.close(fig_grid)

    # ---------- overview (stance-focused line plot) ----------
    fig_ov, ax_ov = plt.subplots(figsize=FIGSIZE)
    # ax_ov.set_title(f'Stance-Focused Foot {component_label} Velocity ({frame_label.replace("_", " ")}) Overview', fontsize=18)
    ax_ov.set_title(f'Stance Foot {component_label} Vel. ({frame_label.replace("_", " ")}, {padding_steps} steps padding) Overview', fontsize=18)


    for foot_idx, lbl in enumerate(foot_labels):
        if foot_idx >= contact_state_array.shape[1]: continue # Skip if no data for this foot

        stance_segments = compute_stance_segments(contact_state_array[:, foot_idx].astype(bool))
        first_segment_for_foot = True
        for s_start, s_end in stance_segments:
            s_pad_start = max(0, s_start - padding_steps)
            s_pad_end = min(min(len(sim_times), velocity_data.shape[0]), s_end + padding_steps)


            if s_pad_start >= s_pad_end: continue
            if foot_idx >= velocity_data.shape[1]: continue


            times_to_plot = sim_times[s_pad_start:s_pad_end]
            data_to_plot = velocity_data[s_pad_start:s_pad_end, foot_idx]

            if times_to_plot.size == 0 or data_to_plot.size == 0: continue
            
            ax_ov.plot(times_to_plot, data_to_plot, label=lbl if first_segment_for_foot else None, alpha=0.7)
            first_segment_for_foot = False
            
    draw_resets(ax_ov, reset_times)
    ax_ov.set_xlabel(r'Time ($\text{s}$)')
    ax_ov.set_ylabel(rf'Velocity {component_label} ($\text{{m}} \cdot \text{{s}}^{{-1}})$')
    ax_ov.legend(ncol=min(len(foot_labels), 4), loc='best')
    
    pdf_ov_path = os.path.join(plot_subdir, f'line_overview_stance_focused.pdf')
    fig_ov.savefig(pdf_ov_path, dpi=600)
    if pickle_subdir: # Check again
        with open(os.path.join(pickle_subdir, f'line_overview_stance_focused.pickle'), 'wb') as f:
            pickle.dump(fig_ov, f)
    plt.close(fig_ov)


def _plot_joint_phase_line_overview(
    data_x: np.ndarray,
    data_y: np.ndarray,
    foot_labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (20, 20)
):
    """Generates an overview line plot for joint phase data."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(title, fontsize=28)

    for i, lbl in enumerate(foot_labels):
        ax.plot(data_x[:, i], data_y[:, i], label=lbl, alpha=0.7)

    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "overview_line.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')
    
    if pickle_dir:
        os.makedirs(pickle_dir, exist_ok=True)
        with open(os.path.join(pickle_dir, "overview_line.pickle"), "wb") as f:
            pickle.dump(fig, f)
    plt.close(fig)

def _plot_joint_phase_line_grid(
    data_x: np.ndarray,
    data_y: np.ndarray,
    foot_labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (20, 20)
):
    """Generates a 2x2 grid of line plots for joint phase data."""
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=30)

    for i, (ax, lbl) in enumerate(zip(axes.flat, foot_labels)):
        ax.plot(data_x[:, i], data_y[:, i], alpha=0.7)
        ax.set_title(lbl, fontsize=26)
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "grid_line.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')

    if pickle_dir:
        os.makedirs(pickle_dir, exist_ok=True)
        with open(os.path.join(pickle_dir, "grid_line.pickle"), "wb") as f:
            pickle.dump(fig, f)
    plt.close(fig)

def _plot_joint_phase_line_single(
    data_x: np.ndarray,
    data_y: np.ndarray,
    foot_label: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (20, 20)
):
    """Generates a single line plot for joint phase data for one foot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(f"{title} - {foot_label}", fontsize=28)
    ax.plot(data_x, data_y, alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)

    safe_lbl = foot_label.replace(" ", "_").lower()
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"single_{safe_lbl}_line.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')
    
    if pickle_dir:
        os.makedirs(pickle_dir, exist_ok=True)
        with open(os.path.join(pickle_dir, f"single_{safe_lbl}_line.pickle"), "wb") as f:
            pickle.dump(fig, f)
    plt.close(fig)


def _plot_joint_phase_hexbin_grid_4x3(
    data_x: np.ndarray,
    data_y: np.ndarray,
    joint_names: list[str],
    leg_row: list[int],
    leg_col: list[int],
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (24, 32),
    gridsize: int = 100
):
    """Generates a 4x3 grid of hexbin plots for joint phase data."""
    fig, axes = plt.subplots(4, 3, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=30)

    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        ax = axes[row, col]
        
        x = data_x[:, j]
        y = data_y[:, j]

        ax.set_title(jn, fontsize=26)
        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap='viridis', mincnt=1)
        counts = hb.get_array()
        
        cbar_label = "Occupancy (samples)"
        if counts.size > 0 and counts.max() > 10 and (counts.max() / (counts.min() + 1e-9) > 20):
            hb.set_norm(colors.LogNorm(vmin=max(1, counts.min()), vmax=counts.max()))
            cbar_label = r"$\log_{10}(\text{Occupancy})$"

        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label(cbar_label, size=22)
        cb.ax.tick_params(labelsize=20)

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "grid_hexbin.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')

    if pickle_dir:
        os.makedirs(pickle_dir, exist_ok=True)
        with open(os.path.join(pickle_dir, "grid_hexbin.pickle"), "wb") as f:
            pickle.dump(fig, f)
    plt.close(fig)

def _plot_joint_phase_line_grid_4x3(
    data_x: np.ndarray,
    data_y: np.ndarray,
    joint_names: list[str],
    leg_row: list[int],
    leg_col: list[int],
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: str,
    pickle_dir: str,
    FIGSIZE: tuple[int, int] = (24, 32)
):
    """Generates a 4x3 grid of line plots for joint phase data."""
    fig, axes = plt.subplots(4, 3, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=30)

    for j, jn in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        ax = axes[row, col]
        
        x = data_x[:, j]
        y = data_y[:, j]

        ax.set_title(jn, fontsize=26)
        ax.plot(x, y, alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "grid_line.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')

    if pickle_dir:
        os.makedirs(pickle_dir, exist_ok=True)
        with open(os.path.join(pickle_dir, "grid_line.pickle"), "wb") as f:
            pickle.dump(fig, f)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# Main plot generation orchestrator
# ----------------------------------------------------------------------------------------------------------------------
def generate_plots(data, output_dir, interactive=False, foot_vel_height_threshold: float = 0.1):
    """
    Recreate all plots from loaded data.
    - data: numpy.lib.npyio.NpzFile containing arrays
    - metrics: dict loaded from metrics_summary.json
    """
    os.makedirs(output_dir, exist_ok=True)
    # pickle_dir = os.path.join(output_dir, "plot_figures_serialized")
    # Ensure pickle_dir is "" or a valid path. If you want to enable pickling, set it here.
    pickle_dir = ""
    # if pickle_dir != "" and not os.path.exists(pickle_dir): # Check if it's a non-empty string before creating
    #     os.makedirs(pickle_dir, exist_ok=True)

    start_time = time.time()

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
    base_linear_velocities = data['base_linear_velocity_body_array']
    base_angular_velocities = data['base_angular_velocity_body_array']
    base_commanded_velocities = data['commanded_velocity_array']
    energy_per_joint, combined_energy, cost_of_transport_time_series = compute_energy_arrays(power_array=power_array, base_lin_vel=base_linear_velocities, reset_steps=reset_timesteps, step_dt=step_dt, robot_mass=total_robot_mass)
    foot_positions_body_frame    = data['foot_positions_body_frame_array'] # (T,4,3)
    foot_positions_contact_frame = data['foot_positions_contact_frame_array'] # (T,4,3)
    foot_positions_world_frame   = data['foot_positions_world_frame_array'] # (T,4,3)
    foot_velocities_world_frame = data['foot_velocities_world_frame_array']
    foot_velocities_body_frame_array = data['foot_velocities_body_frame_array']
    T = base_orientations.shape[0]
    # Calculate magnitudes robustly, T is defined earlier
    if T > 0:
        if foot_velocities_world_frame.ndim == 3 and foot_velocities_world_frame.shape[0] == T: # (T, 4, 3)
            foot_velocities_world_magnitude = np.linalg.norm(foot_velocities_world_frame, axis=2) # (T, 4)
            foot_velocities_body_magnitude = np.linalg.norm(foot_velocities_body_frame_array, axis=2) # (T, 4)
        elif foot_velocities_world_frame.ndim == 2 and T == 1 and foot_velocities_world_frame.shape[0] == 4 and foot_velocities_world_frame.shape[1] == 3: # Single timestep data (4,3)
            foot_velocities_world_magnitude = np.linalg.norm(foot_velocities_world_frame, axis=1).reshape(1, 4) # (1,4)
            foot_velocities_body_magnitude = np.linalg.norm(foot_velocities_body_frame_array, axis=1).reshape(1,4) # (1,4)
        else: # Fallback for unexpected shapes, assuming T might be number of samples
            num_samples = foot_velocities_world_frame.shape[0] if foot_velocities_world_frame.ndim > 1 else 0
            foot_velocities_world_magnitude = np.zeros((num_samples, 4))
            foot_velocities_body_magnitude = np.zeros((num_samples, 4))
            if num_samples > 0 : print(f"[Warning] Unexpected shape for foot_velocities_world_frame: {foot_velocities_world_frame.shape}, T={T}. Magnitudes zeroed.")
    else: # T == 0
        foot_velocities_world_magnitude = np.zeros((0,4))
        foot_velocities_body_magnitude = np.zeros((0,4))
    foot_heights_body_frame = foot_positions_body_frame[:, :, 2]
    foot_heights_contact_frame = foot_positions_contact_frame[:, :, 2]
    step_heights = compute_swing_heights(contact_state=contact_state_array, foot_heights_contact=foot_heights_contact_frame, reset_steps=reset_timesteps, foot_labels=foot_labels)
    step_lengths = compute_swing_lengths(contact_state=contact_state_array, foot_positions_world=foot_positions_world_frame, reset_steps=reset_timesteps, foot_labels=foot_labels)
    swing_durations = compute_swing_durations(contact_state=contact_state_array, sim_env_step_dt=step_dt, foot_labels=foot_labels)

    # build two look-up tables
    leg_row: list = [None] * len(joint_names) # index 0-3
    leg_col: list = [None] * len(joint_names) # index 0-2
    foot_from_joint: list = [None] * len(joint_names)

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
        'position': r'$\text{rad}$',
        'velocity':	r'$\text{rad} \cdot \text{s}^{-1}$',
        'acceleration': r'$\text{rad} \cdot \text{s}^{-2}$',
        'torque': r'$\text{N} \cdot \text{m}$',
        'action_rate': r'$\text{rad} \cdot \text{s}^{-1}$',
        'energy': r'$\text{J}$',
        'combined_energy': r'$\text{J}$',
        'cost_of_transport': r'$-$',
        'power': r'$\text{W}$'
    }

    # Helper for mapping joints to feet
    foot_from_joint = []
    for name in joint_names:
        if   name.startswith('FL_'): foot_from_joint.append(0)
        elif name.startswith('FR_'): foot_from_joint.append(1)
        elif name.startswith('RL_') or name.startswith('HL_'): foot_from_joint.append(2)
        elif name.startswith('RR_') or name.startswith('HR_'): foot_from_joint.append(3)
        else: foot_from_joint.append(None)

    # --- Prepare for Joint Phase Plots ---
    joint_type_map = {0: "hip", 1: "thigh", 2: "calf"}
    joint_indices_by_leg_and_type = [[None] * 3 for _ in range(4)]
    for j, name in enumerate(joint_names):
        row, col = leg_row[j], leg_col[j]
        if row is not None and col is not None:
            joint_indices_by_leg_and_type[row][col] = j

    phase_plot_combinations = [
        # Positions
        {"x_type": "thigh", "y_type": "calf", "data_key": "position"},
        {"x_type": "thigh", "y_type": "hip", "data_key": "position"},
        {"x_type": "hip", "y_type": "calf", "data_key": "position"},
        # Velocities
        {"x_type": "thigh", "y_type": "calf", "data_key": "velocity"},
        {"x_type": "thigh", "y_type": "hip", "data_key": "velocity"},
        {"x_type": "hip", "y_type": "calf", "data_key": "velocity"},
    ]

    data_arrays_for_phase_plots = {
        "position": joint_positions,
        "velocity": joint_velocities,
    }
    unit_labels_for_phase_plots = {
        "position": r"$\text{rad}$",
        "velocity": r"$\text{rad} \cdot \text{s}^{-1}$",
    }
    # --- End Prepare for Joint Phase Plots ---

    futures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 1) Foot contact-force per-foot grid
        futures.append(
            executor.submit(
                _plot_foot_contact_force_per_foot,
                sim_times, contact_forces_array, foot_labels,
                contact_state_array, reset_times, constraint_bounds,
                output_dir, pickle_dir
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
                    output_dir, pickle_dir
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
                    _plot_hist_joint_metric_overview,
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
                output_dir, pickle_dir, FIGSIZE
            )
        )

        futures.append(
            executor.submit(
                _plot_reward_time_series,
                sim_times, reward_array, reset_times,
                output_dir, pickle_dir, FIGSIZE
            )
        )

        futures.append(
            executor.submit(
                _plot_cumulative_reward,
                sim_times, reward_array, reset_times,
                output_dir, pickle_dir, FIGSIZE
            )
        )

        # Instantaneous cost of transport over time
        futures.append(
            executor.submit(
                _plot_cost_of_transport,
                sim_times, cost_of_transport_time_series, reset_times,
                output_dir, pickle_dir, FIGSIZE
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
        # for label, durations in swing_durations.items():
            # futures.append(
            #     executor.submit(
            #         _plot_hist_air_time_per_foot_single,
            #         label, durations, output_dir, pickle_dir, FIGSIZE
            #     )
            # )

        # Base plots
        futures.append(
            executor.submit(
                _plot_combined_base_position,
                sim_times, base_positions, reset_times,
                output_dir, pickle_dir, FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_combined_orientation,
                sim_times, base_orientations, reset_times,
                output_dir, pickle_dir, FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_combined_base_velocity,
                sim_times, data['base_linear_velocity_body_array'], base_commanded_velocities,
                reset_times, output_dir, pickle_dir, FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_combined_base_angular_velocities,
                sim_times, data['base_angular_velocity_body_array'], base_commanded_velocities,
                reset_times, output_dir, pickle_dir, FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_total_base_overview,
                sim_times, base_positions, base_orientations,
                base_linear_velocities, base_angular_velocities,
                reset_times, output_dir, pickle_dir, FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_command_abs_error_base_kinematics,
                sim_times,
                data['base_linear_velocity_body_array'],
                data['base_angular_velocity_body_array'],
                base_commanded_velocities,
                reset_times,
                output_dir,
                pickle_dir,
                FIGSIZE
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
            # futures.append(
            #     executor.submit(
            #         _plot_box_joint_grid,
            #         metric_name, data_arr, joint_names, leg_row, leg_col,
            #         metric_to_unit_mapping, output_dir, pickle_dir
            #     )
            # )
            futures.append(
                executor.submit(
                    _plot_box_joint_metric_overview,
                    metric_name, data_arr, joint_names,
                    metric_to_unit_mapping, output_dir, pickle_dir, FIGSIZE
                )
            )

        # b) Contact-force box plots
        # futures.append(
        #     executor.submit(
        #         _plot_box_contact_forces_grid,
        #         contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE
        #     )
        # )
        futures.append(
            executor.submit(
                _plot_box_contact_forces_overview,
                contact_forces_array, foot_labels, output_dir, pickle_dir, FIGSIZE
            )
        )

        # c) Air-time box plots (grid + per-foot)
        # futures.append(
        #     executor.submit(
        #         _plot_box_air_time_per_foot_grid,
        #         swing_durations, foot_labels, output_dir, pickle_dir, FIGSIZE
        #     )
        # )
        # for label, durations in swing_durations.items():
        #     futures.append(
        #         executor.submit(
        #             _plot_box_air_time_per_foot_single,
        #             label, durations, output_dir, pickle_dir, FIGSIZE
        #         )
        #     )

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
            subdir = os.path.join("foot_com_positions_body_frame", f"{axis_label}")

            futures.append(
                executor.submit(
                    _plot_foot_position_time_series,
                    positions_axis, axis_label, 'body_frame',
                    sim_times, foot_labels, contact_state_array, reset_times,
                    output_dir, pickle_dir, FIGSIZE, subdir
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

            # futures.append(
            #     executor.submit(
            #         _plot_box_metric_grid,
            #         metric_dict,
            #         f"Box Plot of Foot {axis_label} Position (body frame, CoM)",
            #         f"{axis_label} (m)", foot_labels,
            #         output_dir, pickle_dir,
            #         subfolder=subdir,
            #         FIGSIZE=FIGSIZE
            #     )
            # )
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
                    output_dir, pickle_dir, FIGSIZE, subdir
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

            # futures.append(
            #     executor.submit(
            #         _plot_box_metric_grid,
            #         metric_dict,
            #         f"Box Plot of Foot {axis_label} Position (contact frame, toe tip)",
            #         f"{axis_label} (m)", foot_labels,
            #         output_dir, pickle_dir,
            #         subfolder=subdir,
            #         FIGSIZE=FIGSIZE
            #     )
            # )
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
                r"Height ($\text{m}$)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_height", FIGSIZE=FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_hist_metric_overview,
                step_heights,
                "Histogram of Step Height (overview)",
                r"Height ($\text{m}$)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_height", FIGSIZE=FIGSIZE
            )
        )
        # futures.append(
        #     executor.submit(
        #         _plot_box_metric_grid,
        #         step_heights,
        #         "Box Plot of Step Height",
        #         r"Height ($\text{m}$)", foot_labels,
        #         output_dir, pickle_dir,
        #         subfolder="step_height", FIGSIZE=FIGSIZE
        #     )
        # )
        futures.append(
            executor.submit(
                _plot_box_metric_overview,
                step_heights,
                "Box Plot of Step Height (overview)",
                r"Height ($\text{m}$)", foot_labels,
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
                r"Step Length ($\text{m}$)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_length", FIGSIZE=FIGSIZE
            )
        )
        futures.append(
            executor.submit(
                _plot_hist_metric_overview,
                step_lengths,
                "Histogram of Step Length (overview)",
                r"Step Length ($\text{m}$)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_length", FIGSIZE=FIGSIZE
            )
        )
        # futures.append(
        #     executor.submit(
        #         _plot_box_metric_grid,
        #         step_lengths,
        #         "Box Plot of Step Length",
        #         r"Step Length ($\text{m}$)", foot_labels,
        #         output_dir, pickle_dir,
        #         subfolder="step_length", FIGSIZE=FIGSIZE
        #     )
        # )
        futures.append(
            executor.submit(
                _plot_box_metric_overview,
                step_lengths,
                "Box Plot of Step Length (overview)",
                r"Step Length ($\text{m}$)", foot_labels,
                output_dir, pickle_dir,
                subfolder="step_length", FIGSIZE=FIGSIZE
            )
        )

        # ---------------- Foot-velocity time series ----------------
        velocities_axes = {'X': 0, 'Y': 1, 'Z': 2}
        # World frame
        for axis_label, axis_idx in velocities_axes.items():
            velocities_axis = foot_velocities_world_frame[:, :, axis_idx]
            metric_dict    = _array_to_metric_dict(velocities_axis, foot_labels)
            subdir = os.path.join("foot_velocities_world_frame", f"{axis_label}")

            futures.append(
                executor.submit(
                    _plot_foot_velocity_time_series,
                    velocities_axis, axis_label, 'world_frame',
                    sim_times, foot_labels, contact_state_array, reset_times,
                    output_dir, pickle_dir, FIGSIZE, subdir
                )
            )
            # for i, lbl in enumerate(foot_labels):
            #     futures.append(
            #         executor.submit(
            #             _plot_foot_velocity_time_series_single,
            #             velocities_axis, axis_label, 'world_frame',
            #             sim_times, lbl, i, contact_state_array, reset_times,
            #             output_dir, pickle_dir, FIGSIZE, subdir
            #         )
            #     )

            futures.append(
                executor.submit(
                    _plot_hist_metric_grid,
                    metric_dict,
                    f"Histogram of Foot {axis_label} Velocity (world frame)",
                    f"Velocity {axis_label} (m/s)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )
            futures.append(
                executor.submit(
                    _plot_hist_metric_overview,
                    metric_dict,
                    f"Histogram of Foot {axis_label} Velocity (world frame) overview",
                    f"Velocity {axis_label} (m/s)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )

            # futures.append(
            #     executor.submit(
            #         _plot_box_metric_grid,
            #         metric_dict,
            #         f"Box Plot of Foot {axis_label} Velocity (world frame)",
            #         f"Velocity {axis_label} (m/s)", foot_labels,
            #         output_dir, pickle_dir,
            #         subfolder=subdir,
            #         FIGSIZE=FIGSIZE
            #     )
            # )
            futures.append(
                executor.submit(
                    _plot_box_metric_overview,
                    metric_dict,
                    f"Box Plot of Foot {axis_label} Velocity (world frame) overview",
                    f"Velocity {axis_label} (m/s)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )

        # Body frame
        for axis_label, axis_idx in velocities_axes.items():
            velocities_axis = foot_velocities_body_frame_array[:, :, axis_idx]
            metric_dict    = _array_to_metric_dict(velocities_axis, foot_labels)
            subdir = os.path.join("foot_velocities_body_frame", f"{axis_label}")

            futures.append(
                executor.submit(
                    _plot_foot_velocity_time_series,
                    velocities_axis, axis_label, 'body_frame',
                    sim_times, foot_labels, contact_state_array, reset_times,
                    output_dir, pickle_dir, FIGSIZE, subdir
                )
            )
            # for i, lbl in enumerate(foot_labels):
            #     futures.append(
            #         executor.submit(
            #             _plot_foot_velocity_time_series_single,
            #             velocities_axis, axis_label, 'body_frame',
            #             sim_times, lbl, i, contact_state_array, reset_times,
            #             output_dir, pickle_dir, FIGSIZE, subdir
            #         )
            #     )

            futures.append(
                executor.submit(
                    _plot_hist_metric_grid,
                    metric_dict,
                    f"Histogram of Foot {axis_label} Velocity (body frame)",
                    f"Velocity {axis_label} (m/s)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )
            futures.append(
                executor.submit(
                    _plot_hist_metric_overview,
                    metric_dict,
                    f"Histogram of Foot {axis_label} Velocity (body frame) overview",
                    f"Velocity {axis_label} (m/s)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )

            # futures.append(
            #     executor.submit(
            #         _,
            #         metric_dict,
            #         f"Box Plot of Foot {axis_label} Velocity (body frame)",
            #         f"Velocity {axis_label} (m/s)", foot_labels,
            #         output_dir, pickle_dir,
            #         subfolder=subdir,
            #         FIGSIZE=FIGSIZE
            #     )
            # )
            futures.append(
                executor.submit(
                    _plot_box_metric_overview,
                    metric_dict,
                    f"Box Plot of Foot {axis_label} Velocity (body frame) overview",
                    f"Velocity {axis_label} (m/s)", foot_labels,
                    output_dir, pickle_dir,
                    subfolder=subdir,
                    FIGSIZE=FIGSIZE
                )
            )

        # ---------------- Foot-velocity magnitude time series ----------------
        # World frame magnitude
        subdir_world_mag = os.path.join("foot_velocities_world_frame", "magnitude")
        futures.append(
            executor.submit(
                _plot_foot_velocity_magnitude_time_series,
                foot_velocities_world_magnitude, 'world_frame',
                sim_times, foot_labels, contact_state_array, reset_times,
                output_dir, pickle_dir, FIGSIZE, subdir_world_mag
            )
        )
        # for i, lbl in enumerate(foot_labels):
        #     futures.append(
        #         executor.submit(
        #             _plot_foot_velocity_magnitude_time_series_single,
        #             foot_velocities_world_magnitude, 'world_frame',
        #             sim_times, lbl, i, contact_state_array, reset_times,
        #             output_dir, pickle_dir, FIGSIZE, subdir_world_mag
        #         )
        #     )
        metric_dict_world_mag = _array_to_metric_dict(foot_velocities_world_magnitude, foot_labels)
        futures.append(executor.submit(_plot_hist_metric_grid, metric_dict_world_mag, "Histogram of Foot Velocity Magnitude (world frame)", r"Velocity Magnitude ($\text{m} \cdot \text{s}^{-1}$)", foot_labels, output_dir, pickle_dir, subdir_world_mag, FIGSIZE))
        futures.append(executor.submit(_plot_hist_metric_overview, metric_dict_world_mag, "Histogram of Foot Velocity Magnitude (world frame) overview", r"Velocity Magnitude ($\text{m} \cdot \text{s}^{-1}$)", foot_labels, output_dir, pickle_dir, subdir_world_mag, FIGSIZE))
        # futures.append(executor.submit(_plot_box_metric_grid, metric_dict_world_mag, "Box Plot of Foot Velocity Magnitude (world frame)", r"Velocity Magnitude ($\text{m} \cdot \text{s}^{-1}$)", foot_labels, output_dir, pickle_dir, subdir_world_mag, FIGSIZE))
        futures.append(executor.submit(_plot_box_metric_overview, metric_dict_world_mag, "Box Plot of Foot Velocity Magnitude (world frame) overview", r"Velocity Magnitude ($\text{m} \cdot \text{s}^{-1}$)", foot_labels, output_dir, pickle_dir, subdir_world_mag, FIGSIZE))

        # Body frame magnitude
        subdir_body_mag = os.path.join("foot_velocities_body_frame", "magnitude")
        futures.append(
            executor.submit(
                _plot_foot_velocity_magnitude_time_series,
                foot_velocities_body_magnitude, 'body_frame',
                sim_times, foot_labels, contact_state_array, reset_times,
                output_dir, pickle_dir, FIGSIZE, subdir_body_mag
            )
        )
        # for i, lbl in enumerate(foot_labels):
        #     futures.append(
        #         executor.submit(
        #             _plot_foot_velocity_magnitude_time_series_single,
        #             foot_velocities_body_magnitude, 'body_frame',
        #             sim_times, lbl, i, contact_state_array, reset_times,
        #             output_dir, pickle_dir, FIGSIZE, subdir_body_mag
        #         )
        #     )
        metric_dict_body_mag = _array_to_metric_dict(foot_velocities_body_magnitude, foot_labels)
        futures.append(executor.submit(_plot_hist_metric_grid, metric_dict_body_mag, "Histogram of Foot Velocity Magnitude (body frame)", r"Velocity Magnitude ($\text{m} \cdot \text{s}^{-1}$)", foot_labels, output_dir, pickle_dir, subdir_body_mag, FIGSIZE))
        futures.append(executor.submit(_plot_hist_metric_overview, metric_dict_body_mag, "Histogram of Foot Velocity Magnitude (body frame) overview", r"Velocity Magnitude ($\text{m} \cdot \text{s}^{-1}$)", foot_labels, output_dir, pickle_dir, subdir_body_mag, FIGSIZE))
        # futures.append(executor.submit(_plot_box_metric_grid, metric_dict_body_mag, "Box Plot of Foot Velocity Magnitude (body frame)", r"Velocity Magnitude ($\text{m} \cdot \text{s}^{-1}$)", foot_labels, output_dir, pickle_dir, subdir_body_mag, FIGSIZE))
        futures.append(executor.submit(_plot_box_metric_overview, metric_dict_body_mag, "Box Plot of Foot Velocity Magnitude (body frame) overview", r"Velocity Magnitude ($\text{m} \cdot \text{s}^{-1}$)", foot_labels, output_dir, pickle_dir, subdir_body_mag, FIGSIZE))


        foot_heatmap_gridsize = 50 # 2.5cmx2.5cm

        futures.append(
            executor.submit(
                _plot_body_frame_foot_position_heatmap_grid,
                foot_positions_body_frame,
                foot_labels,
                output_dir,
                pickle_dir,
                gridsize=foot_heatmap_gridsize,
                FIGSIZE=(24, 24),
            )
        )

        # for idx, lbl in enumerate(foot_labels):
        #     futures.append(
        #         executor.submit(
        #             _plot_body_frame_foot_position_heatmap_single,
        #             foot_positions_body_frame,
        #             idx,
        #             lbl,
        #             output_dir,
        #             pickle_dir,
        #             gridsize=foot_heatmap_gridsize,
        #             FIGSIZE=(20, 20),
        #         )
        #     )

        futures.append(
            executor.submit(
                _plot_body_frame_foot_position_heatmap,
                foot_positions_body_frame,
                output_dir,
                pickle_dir,
                gridsize=foot_heatmap_gridsize,
                FIGSIZE=(20, 20),
            )
        )

        # Foot XY trajectory plots
        futures.append(
            executor.submit(
                _plot_body_frame_foot_position_xy_grid,
                foot_positions_body_frame,
                foot_labels,
                contact_state_array,
                output_dir,
                pickle_dir,
                FIGSIZE=(20, 20),
            )
        )

        futures.append(
            executor.submit(
                _plot_body_frame_foot_position_xy_overview,
                foot_positions_body_frame,
                foot_labels,
                output_dir,
                pickle_dir,
                FIGSIZE=(20, 20),
            )
        )

        # for idx, lbl in enumerate(foot_labels):
        #     futures.append(
        #         executor.submit(
        #             _plot_body_frame_foot_position_xy_single,
        #             foot_positions_body_frame,
        #             idx,
        #             lbl,
        #             contact_state_array,
        #             output_dir,
        #             pickle_dir,
        #             FIGSIZE=(20, 20),
        #         )
        #     )

        # Foot Velocity vs Height plots
        futures.append(
            executor.submit(
                _plot_foot_velocity_vs_height_grid,
                foot_velocities_world_frame,
                foot_positions_contact_frame,
                foot_labels,
                output_dir,
                pickle_dir,
                foot_vel_height_threshold,
                FIGSIZE,
            )
        )
        futures.append(
            executor.submit(
                _plot_foot_velocity_vs_height_overview,
                foot_velocities_world_frame,
                foot_positions_contact_frame,
                foot_labels,
                output_dir,
                pickle_dir,
                foot_vel_height_threshold,
                FIGSIZE,
            )
        )
        # for idx, lbl in enumerate(foot_labels):
        #     futures.append(
        #         executor.submit(
        #             _plot_foot_velocity_vs_height_single,
        #             foot_velocities_world_frame,
        #             foot_positions_contact_frame,
        #             idx,
        #             lbl,
        #             output_dir,
        #             pickle_dir,
        #             foot_vel_height_threshold,
        #             FIGSIZE,
        #         )
        #     )

        # For comparing to new parallel version
        # futures.append(executor.submit(_animate_body_frame_foot_positions, foot_positions_body_frame, contact_state_array, foot_labels, os.path.join(output_dir, "foot_com_positions_body_frame", "OLD_FOR_COMPARISON_foot_com_positions_body_frame_animation.mp4"), fps=50))

        # --- Joint Phase Plots ---
        for combo in phase_plot_combinations:
            x_type = combo["x_type"]
            y_type = combo["y_type"]
            data_key = combo["data_key"]

            x_col_idx = next(k for k, v in joint_type_map.items() if v == x_type)
            y_col_idx = next(k for k, v in joint_type_map.items() if v == y_type)

            source_data_array = data_arrays_for_phase_plots[data_key]
            num_timesteps = source_data_array.shape[0]
            
            data_x = np.zeros((num_timesteps, 4))
            data_y = np.zeros((num_timesteps, 4))

            for i in range(4):  # For each leg
                x_joint_idx = joint_indices_by_leg_and_type[i][x_col_idx]
                y_joint_idx = joint_indices_by_leg_and_type[i][y_col_idx]
                if x_joint_idx is not None and y_joint_idx is not None:
                    data_x[:, i] = source_data_array[:, x_joint_idx]
                    data_y[:, i] = source_data_array[:, y_joint_idx]

            unit = unit_labels_for_phase_plots[data_key]
            data_name = "Angle" if data_key == "position" else "Ang. Vel."
            
            xlabel = f"{x_type.capitalize()} {data_name} ({unit})"
            ylabel = f"{y_type.capitalize()} {data_name} ({unit})"
            title_prefix = f"{x_type.capitalize()} vs {y_type.capitalize()} Joint {data_name}"
            
            # Directories for Hexbin plots
            hexbin_output_dir = os.path.join(output_dir, "joint_phase_plots", "hexbin", data_key, f"{x_type}_vs_{y_type}")
            hexbin_pickle_dir = os.path.join(pickle_dir, "joint_phase_plots", "hexbin", data_key, f"{x_type}_vs_{y_type}") if pickle_dir else ""

            # Directories for Line plots
            line_output_dir = os.path.join(output_dir, "joint_phase_plots", "line", data_key, f"{x_type}_vs_{y_type}")
            line_pickle_dir = os.path.join(pickle_dir, "joint_phase_plots", "line", data_key, f"{x_type}_vs_{y_type}") if pickle_dir else ""

            # --- Submit Hexbin Plot Jobs ---
            # futures.append(executor.submit(
            #     _plot_joint_phase_overview,
            #     data_x, data_y, foot_labels, xlabel, ylabel, f"{title_prefix} (All Feet, Hexbin)",
            #     hexbin_output_dir, hexbin_pickle_dir, FIGSIZE=(20, 20), gridsize=20
            # ))
            # futures.append(executor.submit(
            #     _plot_joint_phase_grid,
            #     data_x, data_y, foot_labels, xlabel, ylabel, f"{title_prefix} (Hexbin)",
            #     hexbin_output_dir, hexbin_pickle_dir, FIGSIZE=(24, 24), gridsize=20
            # ))
            # for i, label in enumerate(foot_labels):
            #     futures.append(executor.submit(
            #         _plot_joint_phase_single,
            #         data_x[:, i], data_y[:, i], label, xlabel, ylabel, title_prefix,
            #         hexbin_output_dir, hexbin_pickle_dir, FIGSIZE=(20, 20), gridsize=20
            #     ))

            # --- Submit Line Plot Jobs ---
            futures.append(executor.submit(
                _plot_joint_phase_line_overview,
                data_x, data_y, foot_labels, xlabel, ylabel, f"{title_prefix} (All Feet, Line)",
                line_output_dir, line_pickle_dir, FIGSIZE=(20, 20)
            ))
            futures.append(executor.submit(
                _plot_joint_phase_line_grid,
                data_x, data_y, foot_labels, xlabel, ylabel, f"{title_prefix} (Line)",
                line_output_dir, line_pickle_dir, FIGSIZE=(24, 24)
            ))
            # for i, label in enumerate(foot_labels):
            #     futures.append(executor.submit(
            #         _plot_joint_phase_line_single,
            #         data_x[:, i], data_y[:, i], label, xlabel, ylabel, title_prefix,
            #         line_output_dir, line_pickle_dir, FIGSIZE=(20, 20)
            #     ))


        # --- Angle vs. Angular Velocity Phase Plots ---
        av_plot_output_dir = os.path.join(output_dir, "joint_phase_plots", "angle_vs_velocity")
        av_plot_pickle_dir = os.path.join(pickle_dir, "joint_phase_plots", "angle_vs_velocity") if pickle_dir else ""

        av_xlabel = f"Joint Angle ({metric_to_unit_mapping['position']})"
        av_ylabel = f"Joint Angular Velocity ({metric_to_unit_mapping['velocity']})"
        
        # --- Submit Hexbin Plot Jobs ---
        av_hexbin_output_dir = os.path.join(av_plot_output_dir, "hexbin")
        av_hexbin_pickle_dir = os.path.join(av_plot_pickle_dir, "hexbin") if av_plot_pickle_dir else ""
        
        # futures.append(executor.submit(
        #     _plot_joint_phase_hexbin_grid_4x3,
        #     joint_positions, joint_velocities, joint_names, leg_row, leg_col,
        #     av_xlabel, av_ylabel, "Joint Angle vs. Angular Velocity (Hexbin)",
        #     av_hexbin_output_dir, av_hexbin_pickle_dir, gridsize=30
        # ))
        # futures.append(executor.submit(
        #     _plot_joint_phase_overview,
        #     joint_positions, joint_velocities, joint_names,
        #     av_xlabel, av_ylabel, "Joint Angle vs. Angular Velocity (All Joints, Hexbin)",
        #     av_hexbin_output_dir, av_hexbin_pickle_dir, FIGSIZE=(20, 20), gridsize=30
        # ))
        # for j, jn in enumerate(joint_names):
        #     futures.append(executor.submit(
        #         _plot_joint_phase_single,
        #         joint_positions[:, j], joint_velocities[:, j], jn,
        #         av_xlabel, av_ylabel, "Joint Angle vs. Angular Velocity",
        #         av_hexbin_output_dir, av_hexbin_pickle_dir, FIGSIZE=(20, 20), gridsize=30
        #     ))

        # --- Submit Line Plot Jobs ---
        av_line_output_dir = os.path.join(av_plot_output_dir, "line")
        av_line_pickle_dir = os.path.join(av_plot_pickle_dir, "line") if av_plot_pickle_dir else ""
        
        futures.append(executor.submit(
            _plot_joint_phase_line_grid_4x3,
            joint_positions, joint_velocities, joint_names, leg_row, leg_col,
            av_xlabel, av_ylabel, "Joint Angle vs. Angular Velocity (Line)",
            av_line_output_dir, av_line_pickle_dir
        ))
        futures.append(executor.submit(
            _plot_joint_phase_line_overview,
            joint_positions, joint_velocities, joint_names,
            av_xlabel, av_ylabel, "Joint Angle vs. Angular Velocity (All Joints, Line)",
            av_line_output_dir, av_line_pickle_dir
        ))
        # for j, jn in enumerate(joint_names):
        #     futures.append(executor.submit(
        #         _plot_joint_phase_line_single,
        #         joint_positions[:, j], joint_velocities[:, j], jn,
        #         av_xlabel, av_ylabel, "Joint Angle vs. Angular Velocity",
        #         av_line_output_dir, av_line_pickle_dir
        #     ))
        
        # --- New Stance-Only Velocity Plots (World Frame) ---
        padding_for_stance_lineplots = -1#Number of sim steps for padding
        # Consolidated parent directory for all stance-only world frame velocity plots
        stance_plot_parent_dir = "foot_velocities_world_frame_stance_only"

        velocity_components_axes = {'VX': 0, 'VY': 1, 'VZ': 2}
        for comp_label, comp_idx in velocity_components_axes.items():
            # Data for this component: foot_velocities_world_frame (T, 4, 3) -> (T, 4)
            if foot_velocities_world_frame.shape[0] > 0 and foot_velocities_world_frame.ndim == 3:
                current_velocity_component_data = foot_velocities_world_frame[:, :, comp_idx]
            elif foot_velocities_world_frame.shape[0] > 0 and foot_velocities_world_frame.ndim == 2 and foot_velocities_world_frame.shape[1] == 3: # Single step (4,3)
                 current_velocity_component_data = foot_velocities_world_frame[:, comp_idx].reshape(1,4) # Make it (1,4)
            else: # Zero timesteps or unexpected
                current_velocity_component_data = np.zeros((foot_velocities_world_frame.shape[0], 4))

            # Subfolder for this specific component (e.g., foot_velocities_world_frame_stance_only/vx)
            # This is where line, hist, and box plots for this component will go.
            component_specific_plot_dir = os.path.join(stance_plot_parent_dir, comp_label.lower())

            # 1. Line plots (stance-focused)
            futures.append(executor.submit(
                _plot_foot_velocity_time_series_stance_focused, # This function will create its own subfolder using component_label
                sim_times, current_velocity_component_data, comp_label, 'world_frame',
                foot_labels, contact_state_array, reset_times,
                output_dir, pickle_dir, FIGSIZE, stance_plot_parent_dir, padding_for_stance_lineplots
            ))

            # 2. Histograms & Box plots (stance-only)
            stance_only_data_dict = _prepare_stance_only_velocity_data(
                foot_velocities_world_frame, contact_state_array, foot_labels, component_idx=comp_idx
            )
            
            # Histograms
            hist_title = f"Histogram of Stance Foot {comp_label} Velocity (World Frame)"
            hist_xlabel = f"Velocity {comp_label} (m/s)"
            # Pass component_specific_plot_dir as the 'subfolder' argument
            futures.append(executor.submit(_plot_hist_metric_grid, stance_only_data_dict, hist_title, hist_xlabel, foot_labels, output_dir, pickle_dir, component_specific_plot_dir, FIGSIZE))
            futures.append(executor.submit(_plot_hist_metric_overview, stance_only_data_dict, f"{hist_title} Overview", hist_xlabel, foot_labels, output_dir, pickle_dir, component_specific_plot_dir, FIGSIZE))

            # Box plots
            box_title = f"Box Plot of Stance Foot {comp_label} Velocity (World Frame)"
            box_ylabel = f"Velocity {comp_label} (m/s)" # For overview, xlabel is "Foot"
            # futures.append(executor.submit(_plot_box_metric_grid, stance_only_data_dict, box_title, hist_xlabel, foot_labels, output_dir, pickle_dir, component_specific_plot_dir, FIGSIZE))
            futures.append(executor.submit(_plot_box_metric_overview, stance_only_data_dict, f"{box_title} Overview", box_ylabel, foot_labels, output_dir, pickle_dir, component_specific_plot_dir, FIGSIZE))

        # Velocity Magnitude
        comp_label_mag = "Magnitude"
        magnitude_specific_plot_dir = os.path.join(stance_plot_parent_dir, comp_label_mag.lower())

        # 1. Line plots (stance-focused) for Magnitude
        futures.append(executor.submit(
            _plot_foot_velocity_time_series_stance_focused, # This function will create its own subfolder using component_label
            sim_times, foot_velocities_world_magnitude, comp_label_mag, 'world_frame',
            foot_labels, contact_state_array, reset_times,
            output_dir, pickle_dir, FIGSIZE, stance_plot_parent_dir, padding_for_stance_lineplots
        ))

        # 2. Histograms & Box plots (stance-only) for Magnitude
        stance_only_mag_data_dict = _prepare_stance_only_velocity_data(
            foot_velocities_world_magnitude, contact_state_array, foot_labels, component_idx=None # None for magnitude
        )

        # Histograms for Magnitude
        hist_mag_title = f"Histogram of Stance Foot {comp_label_mag} Velocity (World Frame)"
        hist_mag_xlabel = f"Velocity {comp_label_mag} (m/s)"
        futures.append(executor.submit(_plot_hist_metric_grid, stance_only_mag_data_dict, hist_mag_title, hist_mag_xlabel, foot_labels, output_dir, pickle_dir, magnitude_specific_plot_dir, FIGSIZE))
        futures.append(executor.submit(_plot_hist_metric_overview, stance_only_mag_data_dict, f"{hist_mag_title} Overview", hist_mag_xlabel, foot_labels, output_dir, pickle_dir, magnitude_specific_plot_dir, FIGSIZE))

        # Box plots for Magnitude
        box_mag_title = f"Box Plot of Stance Foot {comp_label_mag} Velocity (World Frame)"
        box_mag_ylabel = f"Velocity {comp_label_mag} (m/s)"
        # futures.append(executor.submit(_plot_box_metric_grid, stance_only_mag_data_dict, box_mag_title, hist_mag_xlabel, foot_labels, output_dir, pickle_dir, magnitude_specific_plot_dir, FIGSIZE))
        futures.append(executor.submit(_plot_box_metric_overview, stance_only_mag_data_dict, f"{box_mag_title} Overview", box_mag_ylabel, foot_labels, output_dir, pickle_dir, magnitude_specific_plot_dir, FIGSIZE))


        # This one is not wrapped in a future since it manages its own process pool internally
        if foot_positions_body_frame.shape[0] > 1 : # Avoid animation if only one frame
            _animate_body_frame_pipe_to_ffmpeg(
            foot_positions_body_frame,
            contact_state_array,
            foot_labels,
            os.path.join(output_dir, "foot_com_positions_body_frame", "foot_com_positions_body_frame_animation.mp4"),
            fps=50,
            dpi=300
        )

        # Ensure all tasks complete
        for i, f in enumerate(futures):
            try:
                f.result(timeout=300) # 5 min timeout per plot
            except Exception as e:
                print(f"ERROR: Plot generation for future {i} failed: {e}")
                # Optionally, find out which plot it was if you store more info with futures

    end_time = time.time()
    print(f"Plot generation took {(end_time-start_time):.4f} seconds.")

    # Interactive display if requested
    if interactive:
        plt.ion()
        plt.show()

def main():
    # Set the multiprocessing start method to 'spawn' for cleaner process creation,
    # which helps avoid deadlocks with subprocesses like FFmpeg.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # The start method can only be set once.
        pass

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
    parser.add_argument("--foot_vel_height_threshold", type=float, default=0.1,
                        help="Maximum foot height to include in the foot-velocity-vs-height plot.")
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
        
    generate_plots(data, args.output_dir, interactive=args.interactive, foot_vel_height_threshold=args.foot_vel_height_threshold)

if __name__ == "__main__":
    main()
