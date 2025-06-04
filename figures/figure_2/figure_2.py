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

import scipy.stats as stats


if __name__ == "__main__":

    path_neural_weibull_results = Path(f"../../scripts/results/benchmark/neural4p_weibull_nosmallmegaplots2_basearch6_0b85791_benchmark.csv")    
    path_chao2_results = Path(f"../../scripts/results/benchmark/chao2_estimator_benchmark.csv")    

    
    # Read the data
    df_nw = pd.read_csv(path_neural_weibull_results)
    df_chao2 = pd.read_csv(path_chao2_results)
    
    # Create combined figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))

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
        ax.set_title(f'{dataset.upper()} Dataset')

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
    # ax3.set_ylabel('RMSE')

    plt.tight_layout()
    plt.show()
