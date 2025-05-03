"""
Plotting panels with polygon + associated climate predictors
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
import random 
random.seed(1)

wid = 1
hei = 1
nrows = 5
ncols = 5
inbetween = 0.1

xx = np.arange(0, ncols, (wid+inbetween))
yy = np.arange(0, nrows, (hei+inbetween))

species = range(10)
marker_styles = ['o', 's', '^', 'v', '>', '<', '*', '+', 'x', 'D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_rasters(coords, ax, j):
    polygon = patches.Polygon(coords, closed=True, edgecolor='black', linewidth=2, facecolor = colors[-j])
    ax.add_patch(polygon)
    for xi in xx:
        for yi in yy:
            for (i,offset) in enumerate(np.linspace(0.8, 3.2, 3)):
                sq_coords = [(xi, yi), (xi + wid, yi), (xi + wid, yi + hei), (xi, yi + hei)]
                # if Polygon(coords).intersects(Polygon(sq_coords)):
                if Polygon(coords).contains(Polygon(sq_coords).centroid):
                    sq = patches.Rectangle((xi, yi) +offset, wid, hei, facecolor=colors[i], alpha=0.5, zorder=-i)
                    ax.add_patch(sq)


widp = wid+inbetween
upper_left_coord = [(0, nrows*widp), 
                    (0, (nrows - 2)*widp), 
                    (widp, (nrows - 2)*widp),  
                    (widp, (nrows - 1)*widp), 
                    (3*widp, (nrows-1)*widp), 
                    (3*widp, nrows*widp)]
middle_cords = [(0, 0), 
                (2*widp, 0), 
                # (widp,0),  
                (2*widp, 3*widp), 
                (nrows*widp, 3*widp), 
                (nrows*widp, nrows*widp), 
                ((nrows-2)*widp, nrows*widp), 
                ((nrows-2)*widp, (nrows-1)*widp), 
                (widp, (nrows-1)*widp), 
                (widp, (nrows-2)*widp), 
                (0, (nrows-2)*widp)]
xi, yi = (2*widp, 0)
lower_right_coord = [(xi, yi), (xi + 3*wid, yi), (xi + 3*wid, yi + 3*hei), (xi, yi + 3*hei)]

# plotting only species
fig = plt.figure()
ax = plt.subplot(111, aspect='equal')
coords = upper_left_coord
plot_rasters(coords, ax, 1)
ax.relim()
ax.autoscale_view()
ax.set_axis_off()
fig.savefig("upper_left.png", dpi=300, transparent=True)

fig = plt.figure()
ax = plt.subplot(111, aspect='equal')
coords = middle_cords
plot_rasters(coords, ax, 2)
ax.relim()
ax.autoscale_view()
ax.set_axis_off()
fig.savefig("middle.png", dpi=300, transparent=True)

fig = plt.figure()
ax = plt.subplot(111, aspect='equal')
coords = lower_right_coord
plot_rasters(coords, ax, 3)
ax.relim()
ax.autoscale_view()
ax.set_axis_off()
fig.savefig("lower_right.png", dpi=300, transparent=True)

