import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll
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

# plotting only species

fig = plt.figure()
ax = plt.subplot(111, aspect='equal')
ax.set_xlim((-0.4300000000000001, 9.030000000000001))
ax.set_ylim((-0.4300000000000001, 9.030000000000001))



# Drawing borders for Block 1 (Triangle)
widp = wid+inbetween
upper_left = patches.Polygon([(0, nrows*widp), 
                              (0, (nrows - 2)*widp), 
                              (widp, (nrows - 2)*widp),  
                              (widp, (nrows - 1)*widp), 
                              (3*widp, (nrows-1)*widp), 
                              (3*widp, nrows*widp)], closed=True, edgecolor='black', linewidth=2, facecolor = colors[-1])
ax.add_patch(upper_left)

# Drawing borders for Block 2 (Rectangle)
lower_right = patches.Rectangle((2*widp, 0), 3*widp, 3*widp, edgecolor='black', linewidth=2, facecolor = colors[-2])
ax.add_patch(lower_right)

middle = patches.Polygon([(0, 0), 
                        (2*widp, 0), 
                        # (widp,0),  
                        (2*widp, 3*widp), 
                        (nrows*widp, 3*widp), 
                        (nrows*widp, nrows*widp), 
                        ((nrows-2)*widp, nrows*widp), 
                        ((nrows-2)*widp, (nrows-1)*widp), 
                        (widp, (nrows-1)*widp), 
                        (widp, (nrows-2)*widp), 
                        (0, (nrows-2)*widp)], closed=True, edgecolor='black', linewidth=2, facecolor = colors[-3])

ax.add_patch(middle)
ax.relim()
ax.autoscale_view()
ax.set_axis_off()
fig.savefig("habitats.png", dpi=300, transparent=True)

for xi in xx:
    for yi in yy:
        for (i,offset) in enumerate(np.linspace(0.8, 3.2, 3)):
            sq = patches.Rectangle((xi, yi) +offset, wid, hei, facecolor=colors[i], alpha=0.5, zorder=-i)
            ax.add_patch(sq)


ax.relim()
ax.autoscale_view()
ax.set_axis_off()
fig.savefig("rasters_habitats.png", dpi=300, transparent=True)

# figure of biodiversity
#%%
fig = plt.figure()
ax = plt.subplot(111, aspect='equal')

# Drawing borders for Block 1 (Triangle)
widp = wid+inbetween
upper_left = patches.Polygon([(0, nrows*widp), 
                              (0, (nrows - 2)*widp), 
                              (widp, (nrows - 2)*widp),  
                              (widp, (nrows - 1)*widp), 
                              (3*widp, (nrows-1)*widp), 
                              (3*widp, nrows*widp)], closed=True, edgecolor='black', linewidth=2, facecolor = colors[-1])
ax.add_patch(upper_left)
ax.text(0.5, (nrows-1)*widp, '10', 
            horizontalalignment='center', 
            verticalalignment='center',
            fontsize=30)

# Drawing borders for Block 2 (Rectangle)
lower_right = patches.Rectangle((2*widp, 0), 3*widp, 3*widp, edgecolor='black', fill=True, linewidth=2, facecolor = colors[-2])
ax.add_patch(lower_right)
ax.text(1.5, (nrows-3)*widp, '0', 
            horizontalalignment='center', 
            verticalalignment='center',
            fontsize=30)

middle = patches.Polygon([(0, 0), 
                        (2*widp, 0), 
                        # (widp,0),  
                        (2*widp, 3*widp), 
                        (nrows*widp, 3*widp), 
                        (nrows*widp, nrows*widp), 
                        ((nrows-2)*widp, nrows*widp), 
                        ((nrows-2)*widp, (nrows-1)*widp), 
                        (widp, (nrows-1)*widp), 
                        (widp, (nrows-2)*widp), 
                        (0, (nrows-2)*widp)], closed=True, edgecolor='black', fill=True, linewidth=2, facecolor = colors[-3])

ax.add_patch(middle)
ax.text(3.5*widp, 1.5*widp, '19', 
            horizontalalignment='center', 
            verticalalignment='center',
            fontsize=30)
ax.relim()
ax.autoscale_view()
ax.set_axis_off()
fig.savefig("SR.png", dpi=300, transparent=True)

# %%
