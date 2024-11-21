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
from figure_2_EVA_Copernicus import (
    process_results,
)

# loading
dataset = process_results(path_results = "/home/boussang/SAR_modelling/python/results/EVA_polygons_CHELSA/EVA_Copernicus_CHELSA/EVA_Copernicus_Chelsa_20000.pkl")
df = dataset.gdf
# filtering habitats
# df = df[df.habitat_id.isin(CLASSES)]
df["area_bounding_box"] = [box(*p.bounds).area for p in df.geometry]
df["n_EVA_entries"] = [len(p.geoms) for p in df.geometry]
df["log_area_bounding_box"] = np.log(df["area_bounding_box"])

df = df[df["n_EVA_entries"] > 1] # filtering out only monopoints

# plotting dataset
fig, axs = plt.subplots(1, len(dataset.config["habitats"]), figsize=(15,3), sharey=True)
axs = axs.flatten()
for i,hab in enumerate(dataset.config["habitats"]):
    ax = axs[i]
    gdf = dataset.gdf[dataset.gdf.habitat_id == hab]
    ax.scatter(np.exp(gdf.log_area), np.exp(gdf.log_sr), s=1)
    ax.set_title(hab)
fig.savefig("raw_SAR_EVA_Copernicus.png", dpi=300, transparent=True)

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
fig.savefig("sr_area_EVA_Copernicus.png", dpi=300, transparent=True)


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
fig.savefig("env_features_EVA_Copernicus.png", dpi=300, transparent=True)
