"""
Plotting GBIF number of entries per land cover classes
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from src.data_processing.utils_landcover import CopernicusDataset
import seaborn as sns
import pandas as pd

# importing loading functions from compilation script
sys.path.append(str(Path(__file__).parent / Path("../../../scripts/gbif_processing/")))
from compile_gbif_copernicus_polygons_cinf_no_multipoint import (
    load_and_preprocess_data,
    assign_landcover_types,
)

# importing data
if False:
    gbif_data, climate_raster, lc_raster, lc_dataset = load_and_preprocess_data()
    gbif_data = assign_landcover_types(gbif_data, lc_raster)
    # renaming
    lc_data = CopernicusDataset()
    # legend is loading when loading raster
    lc_data.load_landcover_level3((30, 30, 30.1, 30.1))
    legend = lc_data.legend_l3
    gbif_data["habitat_names"] = gbif_data["Copernicus_landcover_type_id"].replace(legend)

    habitat_counts = gbif_data["habitat_names"].value_counts().sort_values(ascending=False)
    habitat_counts.to_csv("habitat_counts_cinf.csv")
else:
    habitat_counts = pd.read_csv("habitat_counts_cinf.csv")
    

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(y=habitat_counts.habitat_names, x=habitat_counts.nb_entries, palette="viridis", ax=ax)
ax.set_ylabel("Habitat Name")
ax.set_xlabel("Number of occurences")
ax.set_title("Number of occurences per habitat")
ax.set_xscale("log")
# plt.xticks(rotation=90)
fig.savefig(
    "gbif_entries_per_lc_classes.png", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True
)
