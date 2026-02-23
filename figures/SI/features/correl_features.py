""""
Plotting environmental feature correlation structure.'
"""
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

from muscari.plotting import CMAP_BR


SBCV_SAMPLES_PATH = Path(__file__).parent / "../../../data/processed/training_samples/sbcv/a9a058d"
BIOCLIMATE_VARS = [
            "bio1",
            "pet_penman_mean",
            "sfcWind_mean",
            # "bio4",
            # "rsds_1981-2010_range_V.2.1",
            "bio12",
            # "bio15",
        ]

def load_data():
    fold_id = 0
    train_path = SBCV_SAMPLES_PATH / f"fold_{fold_id}_train.parquet"
    # val_path = SBCV_SAMPLES_PATH / f"fold_{fold_id}_val.parquet"
    # test_path = SBCV_SAMPLES_PATH / f"fold_{fold_id}_test.parquet"


    # train_df = gpd.read_parquet(train_path)
    # val_df = gpd.read_parquet(val_path)
    # test_df = gpd.read_parquet(test_path)
    
    # df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    # return df
    return gpd.read_parquet(train_path)
    
if __name__ == "__main__":  
    df = load_data()
    df["log_sp_unit_area"] = np.log(df["sp_unit_area"])
    
    climate_feats = BIOCLIMATE_VARS + [f"std_{v}" for v in BIOCLIMATE_VARS]
    dem_feats = ["elevation", "std_elevation"]

    feature_names = climate_feats + dem_feats + ["log_sp_unit_area"]
    features = df[feature_names]
    
    corr_matrix = features.corr()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap=CMAP_BR, square=True, ax=ax, cbar_kws={'label': 'Correlation', 'ticks': [i/10 for i in range(-10, 11)]}, vmin=-1, vmax=1)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Correlation', size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.savefig("correlation_feature.pdf", transparent=True, dpi=300, bbox_inches='tight')