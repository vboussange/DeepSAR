import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll
import random 

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

pat = []
for xi in xx:
    for yi in yy:
        sq = patches.Rectangle((xi, yi), wid, hei, fill=False)
        ax.add_patch(sq)

random.seed(1)
for _ in range(100):
    xsq = random.choice(xx)
    ysq = random.choice(yy)
    i = random.choice(species)
    x = random.uniform(xsq+0.1, xsq+wid-0.1)
    y = random.uniform(ysq+0.1, ysq+hei-0.1)
    ax.scatter(x,y,marker=marker_styles[i], c=colors[i])

ax.relim()
ax.autoscale_view()
ax.set_axis_off()
fig.savefig("only_species.png", dpi=300, transparent=True)

# plotting random arrangement of plots
Xr = np.random.choice(xx, 4, replace=False)
Yr = np.random.choice(yy, 4, replace=False)


fig = plt.figure()
ax = plt.subplot(111, aspect='equal')

pat = []
for xi in xx:
    for yi in yy:
        sq = patches.Rectangle((xi, yi), wid, hei, fill=False)
        ax.add_patch(sq)
        if (xi,yi) in zip(Xr,Yr):
            sq = patches.Rectangle((xi, yi), wid, hei, fill=True)
            ax.add_patch(sq)

random.seed(1)
for _ in range(100):
    xsq = random.choice(xx)
    ysq = random.choice(yy)
    i = random.choice(species)
    x = random.uniform(xsq+0.1, xsq+wid-0.1)
    y = random.uniform(ysq+0.1, ysq+hei-0.1)
    ax.scatter(x,y,marker=marker_styles[i], c=colors[i])

ax.relim()
ax.autoscale_view()
ax.set_axis_off()
fig.savefig("random_choice.png", dpi=300, transparent=True)

# plotting nested plot arrangement
X = [xx[0], xx[0], xx[1], xx[1]]
Y = [yy[0], yy[1], yy[0], yy[1]]
fig = plt.figure()
ax = plt.subplot(111, aspect='equal')

pat = []
for xi in xx:
    for yi in yy:
        sq = patches.Rectangle((xi, yi), wid, hei, fill=False)
        ax.add_patch(sq)
        if (xi,yi) in zip(X,Y):
            sq = patches.Rectangle((xi, yi), wid, hei, fill=True)
            ax.add_patch(sq)

random.seed(1)
for _ in range(100):
    xsq = random.choice(xx)
    ysq = random.choice(yy)
    i = random.choice(species)
    x = random.uniform(xsq+0.1, xsq+wid-0.1)
    y = random.uniform(ysq+0.1, ysq+hei-0.1)
    ax.scatter(x,y,marker=marker_styles[i], c=colors[i])

ax.relim()
ax.autoscale_view()
ax.set_axis_off()
fig.savefig("nested_choice.png", dpi=300, transparent=True)

# plotting with imshow

gradient = np.linspace(0, 1, 100)
gradient = np.vstack((gradient,) * 100)

fig = plt.figure()
ax = plt.subplot(111, aspect='equal')

# Use imshow with the created axes
im = ax.imshow(gradient, cmap='plasma', extent=[xx[0], xx[-1]+wid, yy[-1]+hei,yy[0]], alpha=0.5)

pat = []
for xi in xx:
    for yi in yy:
        sq = patches.Rectangle((xi, yi), wid, hei, fill=False)
        ax.add_patch(sq)
        if (xi,yi) in zip(Xr,Yr):
            sq = patches.Rectangle((xi, yi), wid, hei, fill=True, alpha=0.3)
            ax.add_patch(sq)

random.seed(1)
for _ in range(100):
    xsq = random.choice(xx)
    ysq = random.choice(yy)
    i = random.choice(species)
    x = random.uniform(xsq+0.1, xsq+wid-0.1)
    y = random.uniform(ysq+0.1, ysq+hei-0.1)
    ax.scatter(x,y,marker=marker_styles[i], c=colors[i])

ax.relim()
ax.autoscale_view()
ax.set_axis_off()
fig.savefig("species_with_gradient.png", dpi=300, transparent=True)
