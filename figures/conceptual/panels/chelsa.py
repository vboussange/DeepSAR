import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll
from matplotlib import path
import random
from matplotlib.patches import Rectangle
from src.data_processing.utils_env_pred import calculate_aggregates, CHELSADataset
from shapely import Polygon, box

# from src.generate_SAR_data_GBIF import generate_random_boxes
random.seed(1)


class RoundedPolygon(patches.PathPatch):
    # https://stackoverflow.com/a/66279687/2912349
    def __init__(self, xy, pad, **kwargs):
        p = path.Path(*self.__round(xy=xy, pad=pad))
        super().__init__(path=p, **kwargs)

    def __round(self, xy, pad):
        n = len(xy)

        for i in range(0, n):

            x0, x1, x2 = np.atleast_1d(xy[i - 1], xy[i], xy[(i + 1) % n])

            d01, d12 = x1 - x0, x2 - x1
            l01, l12 = np.linalg.norm(d01), np.linalg.norm(d12)
            u01, u12 = d01 / l01, d12 / l12

            x00 = x0 + min(pad, 0.5 * l01) * u01
            x01 = x1 - min(pad, 0.5 * l01) * u01
            x10 = x1 + min(pad, 0.5 * l12) * u12
            x11 = x2 - min(pad, 0.5 * l12) * u12

            if i == 0:
                verts = [x00, x01, x1, x10]
            else:
                verts += [x01, x1, x10]

        codes = [path.Path.MOVETO] + n * [
            path.Path.LINETO,
            path.Path.CURVE3,
            path.Path.CURVE3,
        ]

        verts[0] = verts[-1]

        return np.atleast_1d(verts, codes)


wid = 1
hei = 1
nrows = 5
ncols = 5
inbetween = 0.1
cell_size = 2

xx = np.arange(0, ncols, (wid + inbetween))
yy = np.arange(0, nrows, (hei + inbetween))

species = range(10)
marker_styles = ["o", "s", "^", "v", ">", "<", "*", "+", "x", "D"]
colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# plotting only species

fig = plt.figure()
ax = plt.subplot(111, aspect="equal")
# ax.set_xlim((-0.4300000000000001, 9.030000000000001))
# ax.set_ylim((-0.4300000000000001, 9.030000000000001))

env_pred_dataset = CHELSADataset()
CHELSA_arr = env_pred_dataset.load().to_dataset(dim="variable")

# Drawing borders for Block 1 (Triangle)
# widp = wid + inbetween
# lower_right = patches.Rectangle(
#     (0, 0),
#     nrows * widp,
#     nrows * widp,
#     edgecolor="black",
#     fill=True,
#     linewidth=2,
#     facecolor=colors[-1],
# )
# ax.add_patch(lower_right)


# Creating the middle polygon using shapely
middle_coords = [
    (0, 0),
    (nrows * widp, 0),
    # (widp,0),
    # (2*widp, 3*widp),
    # (nrows*widp, 3*widp),
    (nrows * widp, nrows * widp),
    ((nrows - 2) * widp, nrows * widp),
    ((nrows - 2) * widp, (nrows - 1) * widp),
    (widp, (nrows - 1) * widp),
    (widp, (nrows - 4) * widp),
    (0, (nrows - 4) * widp),
]

middle_polygon = Polygon(middle_coords)

# Add the middle polygon to the plot
middle_patch = plt.Polygon(
    np.array(middle_polygon.exterior.coords),
    closed=True,
    edgecolor="black",
    linewidth=2,
    facecolor=colors[-3],
    alpha=0.1
)
ax.add_patch(middle_patch)

ax.relim()
ax.autoscale_view()
# ax.set_axis_off()
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


ax.grid(which="both", color="white", linestyle="--", linewidth=1)
ax.set_xticks(np.arange(-0.1, nrows * widp + 0.1, cell_size))
ax.set_yticks(np.arange(-0.1, nrows * widp + 0.1, cell_size))

# now placing plots
nplots = 40
plots_coords = np.random.randint(0, 3, [nplots, 2])
# Add rectangles
width = 0.2
height = 0.2
for row, col in plots_coords:
    # Alternate color based on row and column
    if (row + col) % 2 == 0:
        color = "tab:red"
    else:
        color = "tab:blue"

    x = col * cell_size + np.random.uniform(0, cell_size - 2 * width)
    y = row * cell_size + np.random.uniform(0, cell_size - 2 * width)

    rect_polygon = box(x, y, x + width, y + height)
    # Check for intersection with middle_polygon before adding
    if rect_polygon.intersects(middle_polygon):
        alpha = 1
    else:
        alpha = 0.1

    rect_patch = plt.Polygon(
        np.array(rect_polygon.exterior.coords),
        closed=True,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
    )
    ax.add_patch(rect_patch)

# placing boxes to make aggregates of plots
rect_patch = RoundedPolygon(
    [(2, 0.5), (2, 3), (6, 3), (6, 0.5)], facecolor="cyan", alpha=0.2, pad=0.4
)
ax.add_patch(rect_patch)

rect_patch = RoundedPolygon(
    [(1, 1), (1, 4), (4, 4), (4, 1)], facecolor="cyan", alpha=0.2, pad=0.4
)
ax.add_patch(rect_patch)

# ax.axis("equal")

chelsa_plot = CHELSA_arr["bio1"].sel(x=slice(32.5,33.5), y=slice(35,34)).to_numpy()
# Superimpose this image onto the plot
ax.imshow(
    chelsa_plot,  # Assuming that chelsa_plot is a 3D array with a single time step
    extent=[0, ncols * widp, 0, nrows * widp],
    origin='lower',
    alpha=0.5,
    cmap='coolwarm'
)

# ax.set_xlim(0, nrows * widp)
fig.savefig("chelsa.png", transparent=True, dpi=300)
