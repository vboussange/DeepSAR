""""
Plotting figure 3 'prediction power of climate, area, and both on SR'
"""
import torch
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
PATH_MLP_TRAINING = Path("../../../scripts/")
sys.path.append(str(Path(__file__).parent / PATH_MLP_TRAINING))
from scripts.train import Config
from eva_chelsa_processing.preprocess_eva_chelsa_megaplots import load_preprocessed_data


if __name__ == "__main__":    
    seed = 1
    MODEL = "large"
    HASH = "71f9fc7"    
    checkpoint_path = PATH_MLP_TRAINING / Path(f"results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")    
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")
    config = results_fit_split_all["config"]
    predictors = results_fit_split_all["all"]["predictors"]
    
    # Load all data to fit PCA globally
    all_gdf = pd.concat([load_preprocessed_data(hab, config.hash_data, config.data_seed) for hab in habitats])
    corr_matrix = all_gdf[predictors].corr()
    
    fig, ax = plt.subplots(figsize=(15, 12))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, ax=ax, cbar_kws={'label': 'Correlation', 'ticks': [i/10 for i in range(-10, 11)]})
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Correlation', size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.savefig("correlation_feature_EVA_EUNIS.png", transparent=True)