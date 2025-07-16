import numpy as np
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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



# Function to create 3D blocks
def plot_vegetation_plots(ax, data):
    x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))

    # Plot 3D bars for each value in the grid
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 1:
                ax.bar3d(i, j, 0., 1, 1, 0.1, color="#344e41", shade=True)
            else:
                ax.bar3d(i, j, 0., 1, 1, 0.1, color="#ced4da", shade=True)

    # Remove all axes, background, and grid
    ax.set_axis_off()
    ax.set_zlim(-0.4, 1)  # Adjust the z-limits based on your data
    ax.view_init(elev=30, azim=225)  
    
fig = plt.figure()
data = np.zeros((40, 40))
# Randomly select a subset of pixels and assign them value 1
np.random.seed(42)  # For reproducibility
n_pixels = 200  # Number of pixels to set to 1
indices = np.random.choice(data.size, size=n_pixels, replace=False)
data.flat[indices] = 1
ax1 = fig.add_subplot(projection='3d')
plot_vegetation_plots(ax1, data)
fig.savefig("vegetation_plots.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=-0.3)

