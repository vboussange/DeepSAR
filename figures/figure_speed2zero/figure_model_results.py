import pandas as pd
import numpy as np
from pathlib import Path
from src.generate_SAR_data_EVA import generate_SAR_data, get_splot_bio_dfs, format_clm5_for_training
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches

habitat = "forest_t1"
data_dir = Path(f"../../../data/data_31_03_2023/{habitat}/")
clm5_df, bio_df = get_splot_bio_dfs(data_dir, 
                                    clm5_stats=["avg", "std"])

# SAR generated data
# this can be generated at each epoch
SAR_data = generate_SAR_data(clm5_df, 
                         bio_df, 
                         npoints=len(clm5_df), 
                         max_aggregate=200, 
                         replace=False, 
                         stats_aggregate=["distance"])
# raw data
raw_data = format_clm5_for_training(clm5_df, stats_aggregate=["mean", "distance"])

def draw_map(ax_map):
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=":")
    ax_map.add_feature(cfeature.LAND, edgecolor="black")
    ax_map.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Overlay the plot locations on the world map
    ax_map.scatter(clm5_df.Longitude, clm5_df.Latitude, c="red", s=1, label="Plot Locations", transform=ccrs.PlateCarree())

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


def draw_single_arrows(ax_scatter, ax_map):
    # Draw arrows from map to scatter plot, raw data
    for i in [110,10]:

        lon, lat = clm5_df.iloc[i].Longitude, clm5_df.iloc[i].Latitude
        a, sr = raw_data.iloc[i].a, raw_data.iloc[i].sr
        
        # Create a connection patch (arrow) between the two axes
        arrow = patches.ConnectionPatch(
            xyA=(lon, lat), coordsA=ax_map.transData,
            xyB=(a, sr), coordsB=ax_scatter.transData,
            arrowstyle='-|>', shrinkA=5, shrinkB=5,
            mutation_scale=20, color='black',
            transform=ax_map.transData
        )

        fig.patches.append(arrow)
        # Circle the point in the scatter plot
        ax_scatter.scatter(a, sr, s=100, facecolors='none', edgecolors='black')
        ax_map.scatter(lon, lat, s=100, facecolors='none', edgecolors='black', marker="s")
    
def draw_multi_arrows(ax_scatter, ax_map):
    SAR_data.sort_values(by='sr', inplace=True)
    nb_arrows = [3, 10]
    # Draw arrows from map to scatter plot (augmented dataset)
    for k,i in enumerate([100, len(SAR_data)-1500]):

        # Coordinates in scatter plot axis
        a, sr = SAR_data.iloc[i].a, SAR_data.iloc[i].sr
        for _ in range(nb_arrows[k]):
            # Coordinates in map axis
            random_row = clm5_df.sample()
            lon, lat = random_row["Longitude"].values[0],random_row["Latitude"].values[0]

            # Create a connection patch (arrow) between the two axes
            arrow = patches.ConnectionPatch(
                xyA=(lon, lat), coordsA=ax_map.transData,
                xyB=(a, sr), coordsB=ax_scatter.transData,
                arrowstyle='-|>', shrinkA=5, shrinkB=5,
                mutation_scale=20, color='black',
                transform=ax_map.transData
            )
            # Add the arrow to the figure
            fig.patches.append(arrow)
            ax_map.scatter(lon, lat, s=100, facecolors='none', edgecolors='black', marker="s")
        ax_scatter.scatter(a, sr, s=100, facecolors='none', edgecolors='black')

def draw_single_arrows_env_cond(ax_scatter, ax_map):
    # Draw arrows from map to scatter plot, raw data
    for i in [110,10]:

        lon, lat = clm5_df.iloc[i].Longitude, clm5_df.iloc[i].Latitude
        a, sr = raw_data.iloc[i].TSOI_10CM_std_mean, raw_data.iloc[i].PRECTmms_avg_mean
        
        # Create a connection patch (arrow) between the two axes
        arrow = patches.ConnectionPatch(
            xyA=(lon, lat), coordsA=ax_map.transData,
            xyB=(a, sr), coordsB=ax_scatter.transData,
            arrowstyle='-|>', shrinkA=5, shrinkB=5,
            mutation_scale=20, color='black',
            transform=ax_map.transData
        )

        fig.patches.append(arrow)
        # Circle the point in the scatter plot
        ax_scatter.scatter(a, sr, s=100, facecolors='none', edgecolors='black')
        ax_map.scatter(lon, lat, s=100, facecolors='none', edgecolors='black', marker="s")
    
# Create a figure and axis with a PlateCarree projection
fig = plt.figure(figsize = (10, 10))
ax_map = fig.add_subplot(224, projection=ccrs.PlateCarree())
draw_map(ax_map)

# Predictors
ax_scatter = fig.add_subplot(223)
ax_scatter.scatter(raw_data.TSOI_10CM_std_mean, raw_data.PRECTmms_avg_mean, label = "Raw data", s = 1.)
ax_scatter.set_ylabel("Precipitation")
ax_scatter.set_xlabel("Temperature")
ax_scatter.legend()
draw_single_arrows_env_cond(ax_scatter, ax_map)

fig.tight_layout()
fig

fig.savefig("raw_data_pred_only.png", dpi = 300, transparent=True)


# SAR
ax_scatter = fig.add_subplot(222)
ax_scatter.set_ylim(raw_data.sr.min(), SAR_data.sr.max())
ax_scatter.set_xlim(raw_data.a.min() - 20, SAR_data.a.max())
draw_scatter_plot(ax_scatter, augmented=False)
draw_single_arrows(ax_scatter, ax_map)

fig.tight_layout()
fig
fig.savefig("raw_data.png", dpi = 300, transparent=True)


# Create a figure and axis with a PlateCarree projection
fig = plt.figure(figsize = (5, 10))
ax_map = fig.add_subplot(212, projection=ccrs.PlateCarree())
draw_map(ax_map)
ax_scatter = fig.add_subplot(211)
draw_scatter_plot(ax_scatter, augmented=True)
ax_scatter.legend()
draw_multi_arrows(ax_scatter, ax_map)
fig.tight_layout()
fig.savefig("augmented_data.png", dpi = 300, transparent=True)
fig



plt.show()


# Draw two arrows that map the location of the two `raw_data` entries to their location in the space `"log(Species richnes)"` x "log(Area)"