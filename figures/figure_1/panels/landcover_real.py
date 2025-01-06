"""
Drawing the land cover panel for conceptutal figure 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import random
import xarray as xr
from src.data_processing.utils_landcover import EUNISDataset

# Set random seed for reproducibility
random.seed(4)

# Load the landcover dataset if required
load_landcover_data = True
if load_landcover_data:
    lc_dataset = EUNISDataset()
    lc_arr = lc_dataset.load_landcover().rio.reproject("EPSG:3035")
    legend = lc_dataset.load_legend()
    inv_legend = {v: k for k, v in legend.items()}
    hab_id = inv_legend["T1"]
    lc_arr = lc_arr.where((150 < lc_arr) & (lc_arr < 170))


class RoundedPolygon(patches.PathPatch):
    """Create a polygon with rounded corners."""
    
    def __init__(self, xy, pad, **kwargs):
        p = mpath.Path(*self.__round(xy=xy, pad=pad))
        super().__init__(path=p, **kwargs)

    def __round(self, xy, pad):
        n = len(xy)
        verts = []

        for i in range(n):
            x0, x1, x2 = np.atleast_1d(xy[i - 1], xy[i], xy[(i + 1) % n])
            d01, d12 = x1 - x0, x2 - x1
            l01, l12 = np.linalg.norm(d01), np.linalg.norm(d12)
            u01, u12 = d01 / l01, d12 / l12

            x00 = x0 + min(pad, 0.5 * l01) * u01
            x01 = x1 - min(pad, 0.5 * l01) * u01
            x10 = x1 + min(pad, 0.5 * l12) * u12
            x11 = x2 - min(pad, 0.5 * l12) * u12

            if i == 0:
                verts.extend([x00, x01, x1, x10])
            else:
                verts.extend([x01, x1, x10])

        codes = [mpath.Path.MOVETO] + n * [mpath.Path.LINETO, mpath.Path.CURVE3, mpath.Path.CURVE3]
        verts[0] = verts[-1]

        return np.atleast_1d(verts), codes

# Grid configuration
width, height = 1, 1
nrows, ncols = 2, 2
inbetween = 0.02

xx = np.arange(0, ncols, width + inbetween)
yy = np.arange(0, nrows, height + inbetween)

# Plotting species
fig, ax = plt.subplots(figsize=(3.5,3.5))
ax.set_aspect('equal')

wid_lc = 8000
for i, xi in enumerate(xx):
    for j, yi in enumerate(yy):
        rect = patches.Rectangle((xi, yi), width, height, fill=False)
        ax.add_patch(rect)
        
        lc_arr_cropped = lc_arr.sel(
            x=slice(4213642 + (i * wid_lc), 4213642 + ((i + 1) * wid_lc)),
            y=slice(2542401 + ((j + 1) * wid_lc), 2542401 + (j * wid_lc))
        ).isnull().to_numpy()[::-1]
        
        G = np.ones((*lc_arr_cropped.shape,4))
        G[lc_arr_cropped] = [0, 0.38, 0, 0.4]
        
        coordsx = np.linspace(xi, xi + width, lc_arr_cropped.shape[0])
        coordsy = np.linspace(yi, yi + height, lc_arr_cropped.shape[1])
        
        ax.imshow(
            G,
            extent=[coordsx[0], coordsx[-1], coordsy[0], coordsy[-1]],
            origin='lower',
            # alpha=0.5,
            # cmap='Greens',
            interpolation='nearest'
        )

        # Adding vegetation plots
        nplots = 20
        for _ in range(nplots):
            plotxi = random.choice(range(1,len(coordsx)-1))
            plotyi = random.choice(range(1,len(coordsy)-1))
            if lc_arr_cropped[plotxi, plotyi]:
                color = "tab:red" #if (i + j) % 3 == 0 else "tab:blue"
                markerfacecolor = color#"white" if not lc_arr_cropped[plotxi, plotyi] else color
            
                ax.scatter(
                    coordsx[plotyi], coordsy[plotxi],
                    marker="x",
                    s = 30,
                    # alpha=0.5,
                    c ="black",
                    # edgecolors="black",
                    # linewidths=4,
                    # markerfacecolor=markerfacecolor, markeredgecolor=color, markersize=5
                )

def add_aggregate_box(p1, p2):
    """Adds a rounded rectangle (aggregate box) to the plot."""
    rect_patch = RoundedPolygon(
        [p1, (p1[0], p2[1]), p2, (p2[0], p1[1])], 
        facecolor=(0,0,0,0.3), pad=0.05, linewidth=3., edgecolor=(0,0,0,1)
    )
    ax.add_patch(rect_patch)


# Adding aggregate boxes
# Bottom left
add_aggregate_box((0.58, 0.2), (0.8, 0.4))
add_aggregate_box((0.45, 0.05), (0.95, 0.5))

# Bottom right
add_aggregate_box((1.1, 0.05), (1.7, 0.8))

# Top right
add_aggregate_box((1.35, 1.7), (1.8, 1.9))

# Top left
add_aggregate_box((0.3, 1.75), (0.85, 1.95))
add_aggregate_box((0.45, 1.65), (0.7, 1.87))

# Create legend handles
# aggregate_patch = RoundedPolygon(
#     [(0, 0), (0, 1), (1, 1), (1, 0)],
#     pad=2,
#     facecolor='azure',
#     alpha=0.2,
#     label='random selection\nof vegetation plots'
# )
# grey_background_patch = patches.Patch(color=[0, 0.4, 0], alpha=1, label='habitat class considered')
# filled_marker = plt.Line2D([0], [0], marker='s', color='w', label='vegetation plot',
#                            markerfacecolor='grey', markersize=5)
# filled_marker = plt.Line2D([0], [0], marker='s', color='w', label='vegetation plot within\nhabitat class considered',
#                            markerfacecolor='white', markeredgecolor='grey', markersize=5)
# nonfilled_marker = plt.Line2D([0], [0], marker='s', color='w', label='vegetation plot',
#                            markerfacecolor='grey', markeredgecolor='grey', markersize=5)
# trained_data = plt.Line2D([0], [0], marker='s', color='w', label='test data',
#                            markerfacecolor='tab:blue', markeredgecolor='tab:blue', markersize=5)
# test_data = plt.Line2D([0], [0], marker='s', color='w', label='train data',
#                            markerfacecolor='tab:red', markeredgecolor='tab:red', markersize=5)
# # Add the legend
# ax.legend(handles=[aggregate_patch, 
#                    grey_background_patch, 
#                    filled_marker, 
#                    nonfilled_marker
#                 #    white_marker
#                    ], loc='center left', bbox_to_anchor=(0.05, 1.3))


ax.set_axis_off()
ax.relim()
ax.autoscale_view()
# fig.supylabel("Vegetation plot", fontsize=18)
fig.tight_layout()

# Save the figure
fig.savefig("landcover_real.pdf", dpi=300, transparent=True)
plt.show()
