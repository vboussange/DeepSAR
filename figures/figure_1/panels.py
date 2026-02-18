import numpy as np
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# Function to create 3D blocks
def plot_3d_env_pred(ax, data, cmap):

    # Plot 3D bars for each value in the grid
    norm = plt.Normalize()
    colors = cmap(norm(data))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.bar3d(i, j, 0, 1, 1, 0.1, color=colors[i, j], shade=True)

    # Remove all axes, background, and grid
    ax.set_axis_off()
    ax.set_zlim(-0.4, 1)  # Adjust the z-limits based on your data
    ax.view_init(elev=30, azim=225)  

fig = plt.figure()
data = np.random.rand(40, 40)
# Add spatial autocorrelation using Gaussian filter
data = ndimage.gaussian_filter(data, sigma=3.)
ax1 = fig.add_subplot(projection='3d')
custom_colors = ["#03045e","#023e8a","#0077b6","#0096c7","#00b4d8","#48cae4","#90e0ef","#ade8f4","#caf0f8"]
custom_cmap = LinearSegmentedColormap.from_list("custom_blue", custom_colors)
plot_3d_env_pred(ax1, data, custom_cmap)
fig.savefig("env_preds1.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=-0.3)


fig = plt.figure()
data = np.random.rand(40, 40)
data = ndimage.gaussian_filter(data, sigma=3.)
ax1 = fig.add_subplot(projection='3d')
custom_colors = ["#2b2d42","#8d99ae","#edf2f4","#ef233c","#d90429"]
custom_cmap = LinearSegmentedColormap.from_list("custom_blue", custom_colors)
plot_3d_env_pred(ax1, data, custom_cmap)
fig.savefig("env_preds2.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=-0.3)



def plot_vegetation_plots(ax, data, green="#54c08aff", gray="#ced4da"):
    # Plot 3D bars for each value in the grid
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 1:
                ax.bar3d(i, j, 0., 1, 1, 0.1, color=green, shade=True)
            else:
                ax.bar3d(i, j, 0., 1, 1, 0.1, color=gray, shade=True)

    # Remove all axes, background, and grid
    ax.set_axis_off()
    ax.set_zlim(-0.4, 1)
    ax.view_init(elev=30, azim=225)


def make_vegetation_data(grid_size, box_size=None, n_pixels=0, seed=42, full=False):
    data = np.zeros((grid_size, grid_size))
    if full:
        data[:, :] = 1
        return data
    if box_size is None:
        return data
    box_size = min(box_size, grid_size)
    start = int((grid_size - box_size) / 2)
    end = start + box_size
    rng = np.random.default_rng(seed)
    available = (end - start) * (end - start)
    n_pixels = min(n_pixels, available)
    indices = rng.choice(available, size=n_pixels, replace=False)
    local = np.zeros((end - start, end - start))
    local.flat[indices] = 1
    data[start:end, start:end] = local
    return data


grid_size = 40
box_size = 24
counts = [60, 160, 320]

for i, n_pixels in enumerate(counts, start=1):
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    data = make_vegetation_data(grid_size, box_size=box_size, n_pixels=n_pixels, seed=40 + i)
    plot_vegetation_plots(ax1, data)
    fig.savefig(f"vegetation_plots_{i}.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=-0.3)

# Full grid (all green)
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
data = make_vegetation_data(grid_size, full=True)
plot_vegetation_plots(ax1, data)
fig.savefig("vegetation_plots_4.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=-0.3)

# Full grid (all green)
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
data = make_vegetation_data(grid_size, grid_size, n_pixels=300, full=False)
plot_vegetation_plots(ax1, data)
fig.savefig("vegetation_plots.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=-0.3)



# Full grid (all green)
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
data = make_vegetation_data(grid_size, grid_size, n_pixels=0, full=False)
plot_vegetation_plots(ax1, data)
fig.savefig("vegetation_plots_empty.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=-0.3)