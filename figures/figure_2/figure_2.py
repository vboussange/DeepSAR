"""
Plotting figure 2 'prediction power of climate, area, and both on SR'
"""
import torch
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error
from src.plotting import boxplot_bypreds
import pandas as pd
from matplotlib.patches import Patch
import geopandas as gpd

import sys
sys.path.append(str(Path(__file__).parent / "../../scripts/"))
from src.neural_4pweibull import initialize_ensemble_model
from train import Config, Trainer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import scipy.stats as stats


if __name__ == "__main__":

    path_neural_weibull_results = Path(f"../../scripts/results/benchmark/neural4p_weibull_nosmallmegaplots2_basearch6_0b85791_benchmark.csv")    
    path_chao2_results = Path(f"../../scripts/results/benchmark/chao2_estimator_benchmark.csv")    

    
    # Read the data
    df_nw = pd.read_csv(path_neural_weibull_results)
    df_chao2 = pd.read_csv(path_chao2_results)
    
    # Create combined figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))

    # First two plots: boxplots for EVA and GIFT datasets
    df_plot = pd.concat([df_nw[(df_nw.train_frac == 1.) & (df_nw.num_params > 4e5)], # TODO: to fix
                         df_chao2], 
                        ignore_index=True)
    metric = "rmse"

    datasets = ['eva', 'gift']
    colors = ['tab:blue', 'tab:orange']
    axes = [ax1, ax2]

    for j, (dataset, ax) in enumerate(zip(datasets, axes)):
        # Define models based on dataset
        if dataset == 'eva':
            models = ["area", "climate", "area+climate"]  # Exclude chao2_estimator for eva
        else:
            models = ["chao2_estimator", "area", "climate", "area+climate"]
        
        box_data = []
        
        for i, model in enumerate(models):
            model_data = df_plot[df_plot['model'] == model]
            metric_col = f"{metric}_{dataset}"
            data = model_data[metric_col].values
            box_data.append(data)

        # Create box plots
        bplot = ax.boxplot(box_data, patch_artist=True, widths=0.6, showfliers=False)
        
        # Add individual data points
        color = colors[j]
        for i, data in enumerate(box_data):
            x = np.random.normal(i + 1, 0.06, size=len(data))  # Add jitter
            ax.scatter(x, data, alpha=0.6, s=10, color=color, zorder=3)

        # Color the boxes
        for patch in bplot['boxes']:
            patch.set_facecolor('none')
            patch.set_edgecolor("none")
        for item in ['caps', 'whiskers']:
            for element in bplot[item]:
                element.set_color("none")
        for element in bplot["medians"]:
            element.set_color("black")

        # Set labels and formatting
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(f'{metric.upper()}') if j == 0 else None
        ax.set_title(f'{dataset.upper()} test dataset')

    # Third plot: errorbar plot for training fractions
    metric = "rmse_gift"
    
    df_plot_train = df_nw[(df_nw['model'] == 'area+climate') & (df_nw['num_params'] > 4e5)]
    train_fracs = sorted(df_plot_train['train_frac'].unique())
    
    means = []
    stds = []
    all_data = []

    for train_frac in train_fracs:
        data = df_plot_train[df_plot_train['train_frac'] == train_frac][metric].values
        means.append(np.mean(data))
        stds.append(np.std(data))
        all_data.append(data)

    # Create errorbar plot
    x_pos = range(1, len(train_fracs) + 1)
    ax3.errorbar(x_pos, means, yerr=stds, fmt='-', capsize=0, capthick=1, 
                 color='black', markersize=6, linewidth=1, zorder=2)

    # Add individual data points
    for i, data in enumerate(all_data):
        x = np.random.normal(i + 1, 0.06, size=len(data))  # Add jitter
        ax3.scatter(x, data, alpha=0.6, s=10, color='tab:purple', zorder=3)

    # Set labels and formatting
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{frac:.0e}' for frac in train_fracs])
    ax3.set_xlabel('Relative nb. of\ntraining samples')
    ax3.set_ylabel('RMSE')

    # Fourth plot: model predictions vs GIFT observations
    # Load model and data for prediction comparison
    gift_data_dir = Path("../../data/processed/GIFT_CHELSA_compilation/6c2d61d/")
    path_results = Path("../../scripts/results/train_seed_1/checkpoint_MSEfit_large_0b85791.pth")
    
    # Load model results
    result_modelling = torch.load(path_results, map_location="cpu")
    
    # Load GIFT dataset
    gift_dataset = gpd.read_parquet(gift_data_dir / "megaplot_data.parquet")
    gift_dataset["log_megaplot_area"] = np.log(gift_dataset["megaplot_area"])
    gift_dataset["log_observed_area"] = np.log(gift_dataset["megaplot_area"])
    gift_dataset = gift_dataset.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    
    # Extract model components
    config = result_modelling["config"]
    predictors = result_modelling["predictors"]
    feature_scaler = result_modelling["feature_scaler"]
    target_scaler = result_modelling["target_scaler"]
    
    # Make predictions
    X = gift_dataset[predictors].copy()
    X = torch.tensor(feature_scaler.transform(X), dtype=torch.float32)
    
    # Initialize model (requires src.neural_4pweibull)
    model = initialize_ensemble_model(result_modelling["ensemble_model_state_dict"], predictors, config, "cpu")
    
    with torch.no_grad():
        y_pred = model(X).numpy()
        y_pred = target_scaler.inverse_transform(y_pred)
    
    gift_dataset["predicted_sr"] = y_pred.squeeze()
    
    # Create inset axes in ax2
    ax4 = inset_axes(ax2, width="40%", height="40%", loc='upper right')
    
    # Plot predictions vs observations
    mask = gift_dataset[["sr", "predicted_sr"]].dropna()
    x = mask["sr"]
    y = mask["predicted_sr"]
    
    ax4.scatter(x, y, alpha=0.6, s=10, color='tab:green')
    
    # Add 1:1 line
    max_val = np.nanmax([x.max(), y.max()])
    min_val = np.nanmin([x.min(), y.min()])
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    
    # Calculate metrics
    # r2_val = stats.pearsonr(x, y)[0]**2
    # rmse_val = np.sqrt(mean_squared_error(x, y))
    
    # Add text with metrics
    # ax4.text(0.05, 0.95, f"R²={r2_val:.2f}\nRMSE={rmse_val:.2f}", 
    #         transform=ax4.transAxes, verticalalignment='top',
    #         bbox=dict(boxstyle="round", fc="w", alpha=0.7))
    
    ax4.set_xlabel('GIFT observed SR', fontsize=8)
    ax4.set_ylabel('Predicted SR', fontsize=8)
    # ax4.set_title('Model vs GIFT\nobservations')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{Path(__file__).stem}.pdf", dpi=300, bbox_inches='tight')
