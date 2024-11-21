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
from figure_2_EVA_EUNIS_null_exp import (
    process_results,
)
habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3"]

# loading
dataset = process_results()
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
fig.savefig("raw_SAR_EVA_EUNIS_null_exp.png", dpi=300, transparent=True)
