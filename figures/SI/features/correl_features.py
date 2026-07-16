"""Plot the environmental feature correlation structure."""
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

from muscari.plotting import CMAP_BR


SBCV_SAMPLES_PATH = Path(__file__).parent / "../../../data/processed/training_samples/sbcv/ceacce0"
BIOCLIMATE_VARS = [
            "bio1",
            "pet_penman_mean",
            "sfcWind_mean",
            "bio12",
        ]

def load_data():
    fold_id = 0
    train_path = SBCV_SAMPLES_PATH / f"fold_{fold_id}_train.parquet"
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
    fig.savefig(
        Path(__file__).with_name("correlation_feature.pdf"),
        transparent=True,
        dpi=300,
        bbox_inches="tight",
    )
