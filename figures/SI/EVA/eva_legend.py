"""
Generates description of EUNIS habitat classes
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from src.data_processing.utils_landcover import CopernicusDataset
from src.data_processing.utils_eva import EVADataset
import seaborn as sns
import pandas as pd

habitats = ["T1", "T3", "S2", "S3", "S6", "R1", "R2", "R3", "R4", "R5", "Q2", "Q5", "Q6"]

# importing data
plot_gdf, dict_sp = EVADataset().load()
single_habs = plot_gdf.drop_duplicates("Level_2")
legend = single_habs[single_habs.Level_2.isin(habitats)][["Level_2", "Level_2_name"]]
legend["Plot number"] = [len(plot_gdf[plot_gdf.Level_2 == hab]) for hab in habitats]
legend.iloc[-1] = ["All", "", len(plot_gdf)]
latex_table = legend.to_latex(index=False)
print(latex_table)

# Save the LaTeX table to a file
with open('legend_EUNIS.tex', 'w') as f:
    f.write(latex_table)