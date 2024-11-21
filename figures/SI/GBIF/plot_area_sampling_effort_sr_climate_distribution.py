"""
Plotting spatial biases
"""
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(str(Path(__file__).parent / Path("../../figure_2/")))
from figure_2_GBIF_Copernicus_cinf import (
    process_results,
)
habitats = [
    " DBF_closed",
    " DBF_open",
    " ENF_closed",
    " ENF_open",
    " bare_sparse_vegetation",
    " herbaceous_vegetation",
    " herbaceous_wetland",
    " shrubland",
]

# loading
dataset = process_results()
df = dataset.gdf
df = df[df.habitat_names.isin(habitats)]

# plotting dataset
fig, axs = plt.subplots(1, len(habitats), figsize=(20,3), sharey=True, sharex=True)
axs = axs.flatten()
for i,hab in enumerate(habitats):
    ax = axs[i]
    gdf = dataset.gdf[dataset.gdf.habitat_names == hab]
    ax.scatter(np.exp(gdf.log_area), np.exp(gdf.log_sr), s=1)
    ax.set_title(hab)
    ax.set_yscale("log")
    ax.set_xscale("log")
axs[0].set_ylabel("SR")
axs[4].set_xlabel("Area (km2)")
fig.tight_layout()
fig.savefig("raw_SAR_GBIF_Copernicus.png", dpi=300, transparent=True)

fig, axs = plt.subplots(1, len(habitats), figsize=(20,3),sharex=True)
axs = axs.flatten()
for i,hab in enumerate(habitats):
    ax = axs[i]
    gdf = dataset.gdf[dataset.gdf.habitat_names == hab]
    ax.scatter(np.exp(gdf.log_area), gdf.std_bio1, s=1)
    ax.set_title(hab)
    # ax.set_yscale("log")
    ax.set_xscale("log")
axs[0].set_ylabel("std_bio1")
axs[4].set_xlabel("Area (km2)")
fig.tight_layout()
fig.savefig("raw_area_std_bio1_GBIF_Copernicus.png", dpi=300, transparent=True)

fig, axs = plt.subplots(1, len(habitats), figsize=(20,3),sharex=True)
axs = axs.flatten()
for i,hab in enumerate(habitats):
    ax = axs[i]
    gdf = dataset.gdf[dataset.gdf.habitat_names == hab]
    ax.scatter(np.exp(gdf.log_area), gdf.std_bio12, s=1)
    ax.set_title(hab)
    # ax.set_yscale("log")
    ax.set_xscale("log")
axs[0].set_ylabel("std_bio12")
axs[4].set_xlabel("Area (km2)")
fig.tight_layout()
fig.savefig("raw_area_std_bio12_GBIF_Copernicus.png", dpi=300, transparent=True)

fig, axs = plt.subplots(1, len(habitats), figsize=(20,3),sharex=True)
axs = axs.flatten()
for i,hab in enumerate(habitats):
    ax = axs[i]
    gdf = dataset.gdf[dataset.gdf.habitat_names == hab]
    ax.scatter(np.exp(gdf.log_area), gdf.bio1, s=1)
    ax.set_title(hab)
    # ax.set_yscale("log")
    ax.set_xscale("log")
axs[0].set_ylabel("bio1")
axs[4].set_xlabel("Area (km2)")
fig.tight_layout()
fig.savefig("raw_area_bio1_GBIF_Copernicus.png", dpi=300, transparent=True)


# plotting area, sampling effort density and sr
fig, axs = plt.subplots(1,3, figsize = (10,4))
ax = axs[0]
g = sns.histplot(df, 
                x = 'log_area', 
                hue="habitat_names", 
                ax=ax, 
                log_scale=(False, False))
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
g.get_legend().remove()

ax = axs[1]
g = sns.histplot(df, x = 'log_sampling_effort_density', hue="habitat_names", ax=ax, log_scale=(False, False))
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
g.get_legend().remove()

ax = axs[2]
sns.histplot(df, x = 'log_sr', hue="habitat_names", ax=ax, log_scale=(False, False))
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
fig.tight_layout()
fig.savefig("sr_area_samplingeffort_GBIF_Copernicus.png", dpi=300, transparent=True)


# plotting bio1 and std_bio1
fig, axs = plt.subplots(2,2, figsize = (10,8))
axs = axs.flatten()
ax = axs[0]
g = sns.histplot(df, 
                x = 'bio1', 
                hue="habitat_names", 
                ax=ax, 
                log_scale=(False, False),
                )
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
g.get_legend().remove()

ax = axs[1]
g = sns.histplot(df, x = 'std_bio1', hue="habitat_names", ax=ax, log_scale=(False, False))
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
g.get_legend().remove()

ax = axs[2]
g = sns.histplot(df, 
                x = 'bio12', 
                hue="habitat_names", 
                ax=ax, 
                log_scale=(False, False),
                )
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
g.get_legend().remove()

ax = axs[3]
g = sns.histplot(df, x = 'std_bio12', hue="habitat_names", ax=ax, log_scale=(False, False))
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
# g.get_legend().remove()

fig.tight_layout()
fig.savefig("env_features_GBIF_Copernicus.png", dpi=300, transparent=True)
