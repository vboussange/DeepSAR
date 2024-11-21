import pandas as pd
import numpy as np
from pathlib import Path
from src.generate_SAR_data_EVA import generate_SAR_data, get_splot_bio_dfs, format_clm5_for_training
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches

def draw_map(ax_map):
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=":")
    ax_map.add_feature(cfeature.LAND, edgecolor="black")
    ax_map.add_feature(cfeature.OCEAN, facecolor="lightblue")


    # Add labels and a legend
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    # ax_map.set_title("Plot Locations")
    ax_map.legend(loc="best")

def draw_scatter_plot(ax_scatter, augmented=False):
    # Add scatterplot
    ax_scatter.scatter(raw_data.a, raw_data.sr, label = "Raw data", s = 1.)
    if augmented:
        ax_scatter.scatter(SAR_data.a, SAR_data.sr, label = "Augmented data", s = 1.)
    ax_scatter.set_yscale("log")
    ax_scatter.set_xscale("log")
    ax_scatter.set_ylabel("log(Species richnes)")
    ax_scatter.set_xlabel("log(Area)")


# Create a figure and axis with a PlateCarree projection
fig = plt.figure(figsize = (10, 5))
ax_map = fig.add_subplot(121, projection=ccrs.PlateCarree())
draw_map(ax_map)
habitats = ["forest_t1", "forest_t3", "grass_r3", "grass_r4", "scrub_s2", "scrub_s6"]
tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
for i,habitat in enumerate(habitats):
    data_dir = Path(f"../../../data/data_31_03_2023/{habitat}/")
    clm5_df, bio_df = get_splot_bio_dfs(data_dir, 
                                        clm5_stats=["avg", "std"])
    # Overlay the plot locations on the world map
    ax_map.scatter(clm5_df.Longitude, clm5_df.Latitude, c=tab_colors[i], s=1, label=habitat, transform=ccrs.PlateCarree())
ax_map.legend()
fig.tight_layout()
fig

ax_scatter = fig.add_subplot(122)

for i,habitat in enumerate(habitats):
    data_dir = Path(f"../../../data/data_31_03_2023/{habitat}/")
    clm5_df, bio_df = get_splot_bio_dfs(data_dir, 
                                        clm5_stats=["avg", "std"])
    raw_data = format_clm5_for_training(clm5_df, stats_aggregate=["mean", "distance"])
    ax_scatter.scatter(raw_data.TSOI_10CM_std_mean, raw_data.PRECTmms_avg_mean, label = habitat, c=tab_colors[i], s = 1.)
ax_scatter.set_ylabel("Precipitation")
ax_scatter.set_xlabel("Temperature")
ax_scatter.legend()
fig.tight_layout()
fig

fig.savefig("EVA_dataset.png", dpi = 300, transparent=True)