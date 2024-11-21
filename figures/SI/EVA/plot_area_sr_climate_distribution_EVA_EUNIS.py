"""
Plotting spatial biases
"""
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import box
import numpy as np

sys.path.append(str(Path(__file__).parent / Path("../../figure_2/")))
from figure_2_EVA_EUNIS import (
    process_results,
)
habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3"]

# loading
dataset = process_results(path_results = "/home/boussang/SAR_modelling/python/results/EVA_polygons_CHELSA/EVA_EUNIS_CHELSA/EVA_EUNIS_Chelsa_20000.pkl")
df = dataset.gdf
# filtering habitats
df = df[df.habitat_id.isin(habitats)]
df["area_bounding_box"] = [box(*p.bounds).area for p in df.geometry]
df["n_EVA_entries"] = [len(p.geoms) for p in df.geometry]
df["log_area_bounding_box"] = np.log(df["area_bounding_box"])

df = df[df["n_EVA_entries"] > 1] # filtering out only monopoints

# plotting dataset
fig, axs = plt.subplots(1, len(habitats), figsize=(15,3), sharey=True)
axs = axs.flatten()
for i,hab in enumerate(habitats):
    ax = axs[i]
    gdf = dataset.gdf[dataset.gdf.habitat_id == hab]
    ax.scatter(np.exp(gdf.log_area), np.exp(gdf.log_sr), s=1)
    ax.set_title(hab)
    ax.set_yscale("log")
    ax.set_xscale("log")
axs[0].set_ylabel("SR")
axs[4].set_xlabel("Area (m2)")
fig.tight_layout()
fig.savefig("raw_SAR_EVA_EUNIS.png", dpi=300, transparent=True)


fig, axs = plt.subplots(1, len(habitats), figsize=(20,3),sharex=True)
axs = axs.flatten()
for i,hab in enumerate(habitats):
    ax = axs[i]
    gdf = dataset.gdf[dataset.gdf.habitat_id == hab]
    ax.scatter(np.exp(gdf.log_area), gdf.std_bio1, s=1)
    ax.set_title(hab)
    # ax.set_yscale("log")
    ax.set_xscale("log")
axs[0].set_ylabel("std_bio1")
axs[4].set_xlabel("Area (km2)")
fig.tight_layout()
fig.savefig("raw_area_std_bio1_EVA_EUNIS.png", dpi=300, transparent=True)

fig, axs = plt.subplots(1, len(habitats), figsize=(20,3),sharex=True)
axs = axs.flatten()
for i,hab in enumerate(habitats):
    ax = axs[i]
    gdf = dataset.gdf[dataset.gdf.habitat_id == hab]
    ax.scatter(np.exp(gdf.log_area), gdf.std_bio12, s=1)
    ax.set_title(hab)
    # ax.set_yscale("log")
    ax.set_xscale("log")
axs[0].set_ylabel("std_bio12")
axs[4].set_xlabel("Area (km2)")
fig.tight_layout()
fig.savefig("raw_area_std_bio12_EVA_EUNIS.png", dpi=300, transparent=True)

fig, axs = plt.subplots(1, len(habitats), figsize=(20,3),sharex=True)
axs = axs.flatten()
for i,hab in enumerate(habitats):
    ax = axs[i]
    gdf = dataset.gdf[dataset.gdf.habitat_id == hab]
    ax.scatter(np.exp(gdf.log_area), gdf.bio1, s=1)
    ax.set_title(hab)
    # ax.set_yscale("log")
    ax.set_xscale("log")
axs[0].set_ylabel("bio1")
axs[4].set_xlabel("Area (km2)")
fig.tight_layout()
fig.savefig("raw_area_bio1_EVA_EUNIS.png", dpi=300, transparent=True)

# plotting area, sampling effort density and sr
fig, axs = plt.subplots(1,4, figsize = (14,4))
ax = axs[0]
g = sns.histplot(df, 
                x = 'area', 
                hue="habitat_id", 
                ax=ax, 
                log_scale=(True, False))
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
g.get_legend().remove()

ax = axs[3]
sns.histplot(df, x = 'area_bounding_box', hue="habitat_id", ax=ax, log_scale=(True, False)) # tofix
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')

ax = axs[1]
g = sns.histplot(df, x = 'sr', hue="habitat_id", ax=ax, log_scale=(True, False))
g.get_legend().remove()
ax.set_ylabel('Frequency')

ax = axs[2]
g = sns.histplot(df, x = 'n_EVA_entries', hue="habitat_id", ax=ax, log_scale=(True, False))
g.get_legend().remove()
ax.set_ylabel('Frequency')

fig.tight_layout()
fig.savefig("sr_area_EVA_EUNIS.png", dpi=300, transparent=True)


# plotting bio1 and std_bio1
fig, axs = plt.subplots(2,2, figsize = (10,8))
axs = axs.flatten()
ax = axs[0]
g = sns.histplot(df, 
                x = 'bio1', 
                hue="habitat_id", 
                ax=ax, 
                log_scale=(False, False),
                )
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
g.get_legend().remove()

ax = axs[1]
g = sns.histplot(df, x = 'std_bio1', hue="habitat_id", ax=ax, log_scale=(False, False), )
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
g.get_legend().remove()

ax = axs[2]
g = sns.histplot(df, 
                x = 'bio12', 
                hue="habitat_id", 
                ax=ax, 
                log_scale=(False, False),
                )
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
g.get_legend().remove()

ax = axs[3]
g = sns.histplot(df, x = 'std_bio12', hue="habitat_id", ax=ax, log_scale=(False, False), )
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Frequency')
# g.get_legend().remove()

fig.tight_layout()
fig.savefig("env_features_EVA_EUNIS.png", dpi=300, transparent=True)
