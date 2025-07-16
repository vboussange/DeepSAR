"""
Plotting training sample size vs model performance
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    path_neural_weibull_results = Path(f"../../../scripts/results/benchmark/neural4p_weibull_nosmallsp_units2_basearch6_0b85791_benchmark.csv")    
    
    # Read the data
    df_nw = pd.read_csv(path_neural_weibull_results)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
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
    # Create box plots
    bplot = ax.boxplot(all_data, patch_artist=True, widths=0.6, showfliers=False)
    
    # Color the boxes (make them invisible except medians)
    for patch in bplot['boxes']:
        patch.set_facecolor('none')
        patch.set_edgecolor("none")
    for item in ['caps', 'whiskers']:
        for element in bplot[item]:
            element.set_color("none")
    for element in bplot["medians"]:
        element.set_color("black")
    
    # Add line connecting medians
    medians = [np.median(data) for data in all_data]
    ax.plot(x_pos, medians, '--', color="#f72585", linewidth=1, zorder=2)
    # Add individual data points
    for i, data in enumerate(all_data):
        x = np.random.normal(i + 1, 0.06, size=len(data))  # Add jitter
        ax.scatter(x, data, alpha=0.6, s=10, color="#f72585", zorder=3)

    # Set labels and formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'$10^{{{int(np.log10(frac))}}}$' if i % 2 == 0 else '' for i, frac in enumerate(train_fracs)])
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('RMSE')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{Path(__file__).stem}.pdf", dpi=300, bbox_inches='tight')
