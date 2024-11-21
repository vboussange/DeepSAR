"""
Plotting EVA number of entries per land cover classes

TODO: WIP
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from src.data_processing.utils_landcover import CopernicusDataset
from src.data_processing.utils_eva import EVADataset
import seaborn as sns
import pandas as pd


# importing data
if False:
    plot_gdf, dict_sp = EVADataset().load()
    # gbif_data = assign_landcover_types(gbif_data, lc_raster)
    habitat_counts = pd.DataFrame(plot_gdf["Level_2"].value_counts().sort_values(ascending=False))
    habitat_counts.to_csv("habitat_counts_cinf.csv")
else:
    habitat_counts = pd.read_csv("habitat_counts_cinf.csv", index_col=0)
    

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(y=habitat_counts.index, x=habitat_counts.Level_2, palette="viridis", ax=ax)
ax.set_ylabel("Habitat Name")
ax.set_xlabel("Number of occurences")
ax.set_title("Number of occurences per habitat")
ax.set_xscale("log")
# plt.xticks(rotation=90)
fig.savefig(
    "EVA_entries_per_lc_classes.png", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True
)
