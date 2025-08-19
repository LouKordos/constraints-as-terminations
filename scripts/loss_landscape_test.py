import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D

# --- 1. Setup the Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
plt.style.use('default')

# --- 2. Define Data for the Loss Landscape ---
x_loss = np.array([0, 2.0, 4.0, 5.0, 6.5, 8.0, 10])
y_loss = np.array([10, 1.0, 5.5, 6.0, 3.0, 6.5, 9.5])

# --- 3. Smooth the Loss Curve ---
def create_smooth_curve(x_pts, y_pts, num_points=300):
    """Generates smooth x and y coordinates from a set of points."""
    t = np.arange(len(x_pts))
    spl_x = make_interp_spline(t, x_pts, k=3)
    spl_y = make_interp_spline(t, y_pts, k=3)
    t_smooth = np.linspace(t.min(), t.max(), num_points)
    x_smooth = spl_x(t_smooth)
    y_smooth = spl_y(t_smooth)
    return x_smooth, y_smooth

x_loss_smooth, y_loss_smooth = create_smooth_curve(x_loss, y_loss)

# --- 4. Generate Curriculum Paths based on the Loss Landscape ---
green_path_offset = 0.6
orange_path_offset = 1.2
threshold_y = 5.5

# Green Path: Stops at the global minimum.
x_green_path = x_loss_smooth[::-1]
y_green_path = y_loss_smooth[::-1] + green_path_offset
min_idx_original = np.argmin(y_loss_smooth)
stop_idx_green = len(y_loss_smooth) - 1 - min_idx_original
x_green_smooth = x_green_path[:stop_idx_green + 1]
y_green_smooth = y_green_path[:stop_idx_green + 1]
green_stop_marker_pos = (x_green_smooth[-1], y_green_smooth[-1])

# Orange Path: Enters the local minimum and stops at the second threshold crossing.
x_orange_path = x_loss_smooth[::-1]
y_orange_path = y_loss_smooth[::-1] + orange_path_offset
crossings = np.where(np.diff(np.sign(y_orange_path - threshold_y)))[0]
if crossings.size >= 2:
    stop_idx_orange = crossings[1]
    x_orange_smooth = x_orange_path[:stop_idx_orange + 1]
    y_orange_smooth = y_orange_path[:stop_idx_orange + 1]
else:
    x_orange_smooth = x_orange_path
    y_orange_smooth = y_orange_path
# Get coordinates for the orange 'X' marker
orange_stop_marker_pos = (x_orange_smooth[-1], y_orange_smooth[-1])


# --- 5. Plot the Main Elements ---
# Decreased line thickness
ax.plot(x_loss_smooth, y_loss_smooth, color='black', linewidth=2.5, zorder=1)
ax.plot(x_green_smooth, y_green_smooth, color='#25792A', linewidth=2.5, zorder=2)
ax.plot(x_orange_smooth, y_orange_smooth, color='#E0731D', linewidth=2.5, zorder=2)
ax.axhline(y=threshold_y, color='#D62728', linestyle='--', linewidth=2.5, zorder=3)

# Add the 'X' markers for both stop points
ax.scatter(green_stop_marker_pos[0], green_stop_marker_pos[1],
           marker='x', color='#08420A', s=200, lw=4, zorder=5)
ax.scatter(orange_stop_marker_pos[0], orange_stop_marker_pos[1],
           marker='x', color='#BF5B14', s=200, lw=4, zorder=5)


# --- 6. Add Arrows to Indicate Path Direction ---
def add_arrows_to_path(ax, x_path, y_path, color, num_arrows=5):
    """Adds directional arrows to a plotted line using mutation_scale."""
    for i in range(1, num_arrows + 1):
        idx = int(len(x_path) * i / (num_arrows + 1.5))
        if idx < 2: continue
        p1 = (x_path[idx - 2], y_path[idx - 2])
        p2 = (x_path[idx], y_path[idx])
        ax.annotate("",
                    xy=p2, xycoords='data',
                    xytext=p1, textcoords='data',
                    arrowprops=dict(arrowstyle="-|>",
                                    mutation_scale=25,
                                    color=color,
                                    lw=2.5,
                                    shrinkA=0, shrinkB=0),
                    zorder=4)

add_arrows_to_path(ax, x_green_smooth, y_green_smooth, color='#25792A', num_arrows=4)
add_arrows_to_path(ax, x_orange_smooth, y_orange_smooth, color='#E0731D', num_arrows=2)

# --- 7. Add Text Annotations and Labels ---
# "Trotting" text moved slightly higher
ax.text(2.0, 2.4, 'Trotting', fontsize=18, fontweight='bold', ha='center')
ax.text(6.5, 2.2, 'Pronking', fontsize=18, fontweight='bold', ha='center')
ax.set_xlabel('Network Parameter (simplified)', fontsize=20, labelpad=10)
ax.set_ylabel('Loss', fontsize=20, labelpad=10)

# --- 8. Create the Custom Legend ---
legend_elements = [
    Line2D([0], [0], color='#E0731D', lw=4, label='Adaptive curriculum'),
    Line2D([0], [0], color='#D62728', lw=3, linestyle='--', label='Adaptive curriculum threshold'),
    Line2D([0], [0], color='#25792A', lw=4, label='Linear curriculum')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=16,
          fancybox=True, shadow=False,
          facecolor='whitesmoke', edgecolor='gray')

# --- 9. Final Styling and Cleanup ---
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0, 11)

# Turn off all the default axes spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Add custom arrows for the x and y axes
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

# Add arrows. clip_on=False allows the arrowheads to extend beyond the plot area.
ax.arrow(xmin, ymin, (xmax - xmin) * 0.99, 0,
         head_width=0.4, head_length=0.4, fc='black', ec='black', lw=2, clip_on=False)
ax.arrow(xmin, ymin, 0, (ymax - ymin) * 0.99,
         head_width=0.25, head_length=0.4, fc='black', ec='black', lw=2, clip_on=False)


plt.tight_layout()

# --- 10. Save the Plot to a PDF file ---
plt.savefig('loss_landscape.pdf', dpi=600, bbox_inches='tight')

print("Plot saved as loss_landscape.pdf")
