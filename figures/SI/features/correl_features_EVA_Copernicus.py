""""
Plotting figure 3 'prediction power of climate, area, and both on SR'
"""

import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from src.model_validation import get_spatial_block_cv_index
from xgboost import XGBRegressor
import shap

from src.data_processing.utils_env_pred import calculate_aggregates, CHELSADataset
from src.data_processing.utils_landcover import CopernicusDataset
from src.data_processing.utils_landcover import CopernicusDataset
import sys

sys.path.append(str(Path(__file__).parent / Path("../../figure_2/")))
from figure_2_EVA_Copernicus import (
    process_results,
)

if __name__ == "__main__":    
    dataset = process_results()
    corr_matrix = dataset.gdf[dataset.aggregate_labels].corr()
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, ax=ax)
    fig.savefig("correlation_feature_EVA_Copernicus.png", transparent=True)