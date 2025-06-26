"""
Plotting figure 2 'prediction power of climate, area, and both on SR'
TODO: change color scheme for plotting.CMAP_BR blue and reds
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
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import MultiComparison
from src.cld import create_comp_matrix_allpair_t_test, multcomp_letters

import scipy.stats as stats

def report_model_performance_and_significance(df_plot, metric, output_file="model_performance_significance.txt"):
    """
    Report model performance and statistical significance for eva and gift datasets.
    
    Parameters:
    -----------
    df_plot : pd.DataFrame
        Combined dataframe with model results
    output_file : str
        Path to output text file for results
    """
    datasets = ['eva', 'gift']
    
    with open(output_file, "w") as file:
        for dataset in datasets:
            metric_col = f"{metric}_{dataset}"
            
            # Get available models for this dataset
            available_models = []
            model_data_dict = {}
            
            for model in df_plot['model'].unique():
                model_data = df_plot[df_plot['model'] == model]
                if not model_data.empty and metric_col in model_data.columns:
                    performance = model_data[metric_col].dropna().values
                    if len(performance) > 0:
                        available_models.append(model)
                        model_data_dict[model] = performance
            
            if not available_models:
                continue
                
            print(f"\n{dataset.upper()} Dataset Performance", file=file)
            print("=" * 50, file=file)
            
            # Performance summary table
            dataset_results = []
            for model in available_models:
                performance = model_data_dict[model]
                dataset_results.append({
                    'Model': model,
                    'RMSE_mean': np.mean(performance),
                    'RMSE_std': np.std(performance),
                    'N': len(performance)
                })
            
            results_df = pd.DataFrame(dataset_results)
            results_df['RMSE'] = results_df.apply(lambda x: f"{x['RMSE_mean']:.4f} ± {x['RMSE_std']:.4f}", axis=1)
            summary_table = results_df[['Model', 'RMSE', 'N']]
            print(summary_table.to_string(index=False), file=file)
            
            # Statistical significance tests (pairwise comparisons)
            print(f"\nPairwise Statistical Significance Tests ({dataset.upper()})", file=file)
            print("-" * 50, file=file)
            
            
            # Create significance matrix
            n_models = len(available_models)
            for i in range(n_models):
                for j in range(i+1, n_models):
                    model1, model2 = available_models[i], available_models[j]
                    data1, data2 = model_data_dict[model1], model_data_dict[model2]
                    
                    # Calculate means for relative difference
                    median1, median2 = np.median(data1), np.median(data2)
                    rel_diff = ((median2 - median1) / median1) * 100
                    
                    # Perform t-test
                    statistic, p_value = ttest_ind(data1, data2)
                    
                    # Determine significance level
                    if p_value < 0.001:
                        sig_level = "***"
                    elif p_value < 0.01:
                        sig_level = "**"
                    elif p_value < 0.05:
                        sig_level = "*"
                    else:
                        sig_level = "ns"
                    
                    print(f"{model1} vs {model2}: t={statistic:.3f}, p={p_value:.4f} {sig_level}, rel_diff={rel_diff:+.1f}%", file=file)
            
            print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant", file=file)


if __name__ == "__main__":

    path_neural_weibull_results = Path(f"../../scripts/results/benchmark/neural4p_weibull_nosmallmegaplots2_basearch6_0b85791_benchmark.csv")    
    path_chao2_results = Path(f"../../scripts/results/benchmark/chao2_estimator_benchmark.csv")    

    
    # Read the data
    df_nw = pd.read_csv(path_neural_weibull_results)
    df_chao2 = pd.read_csv(path_chao2_results)
    
    # Create combined figure

    # First two plots: boxplots for EVA and GIFT datasets
    df_plot = pd.concat([df_nw[(df_nw.train_frac == 1.) & (df_nw.num_params > 4e5)], # TODO: to fix
                         df_chao2], 
                        ignore_index=True)
    # Replace "climate" with "environment" in the model column
    df_plot['model'] = df_plot['model'].str.replace('climate', 'environment')
    metric = "rmse"
    report_model_performance_and_significance(df_plot, metric, output_file="model_performance_significance.txt")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
   
    datasets = ['eva', 'gift']
    colors = ['tab:blue', 'tab:purple']
    
    
    axes = [ax1, ax2]
    

    for j, (dataset, ax) in enumerate(zip(datasets, axes)):
        # Define models based on dataset
        if dataset == 'eva':
            models = ["area", "environment", "area+environment"]  # Exclude chao2_estimator for eva
        else:
            models = ["chao2_estimator", "area", "environment", "area+environment"]
        
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
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(f'{metric.upper()}') if j == 0 else None
        ax.set_title(f'{dataset.upper()} test dataset')
        # Increase y-axis limits by 10%
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        # Statistical significance annotations
        
        alpha = 0.05
        spread = 0.6
        
        # Flatten data and create groups for statistical comparison
        flat_data = []
        group_labels = []
        for i, data in enumerate(box_data):
            flat_data.extend(data)
            group_labels.extend([models[i]] * len(data))
        
        if len(set(group_labels)) > 1:  # Only if we have multiple groups
            mc = MultiComparison(flat_data, group_labels)
            test_results = mc.allpairtest(stats.ttest_ind, alpha=alpha)
            
            comp_matrix = create_comp_matrix_allpair_t_test(test_results)
            letters = multcomp_letters(comp_matrix < alpha)
            
            # Add letter annotations above boxplots
            for i, model in enumerate(models):
                if model in letters:
                    # Calculate whisker position for annotation placement
                    data_vals = box_data[i]
                    q75 = np.percentile(data_vals, 75)
                    iqr = np.percentile(data_vals, 75) - np.percentile(data_vals, 25)
                    whisker_top = q75 + 1.5 * iqr
                    ypos = min(whisker_top, max(data_vals)) + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                    
                    # Get median position
                    median_val = np.median(data_vals)
                    ax.text(i + 0.7, median_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01, 
                            letters[model], ha='left', va='bottom', 
                            fontsize=10, color='black')

    # Third plot: errorbar plot for training fractions
    ax3 = inset_axes(ax1, width="40%", height="40%", loc='upper right', bbox_to_anchor=(-0.05, 0, 1, 1), bbox_transform=ax1.transAxes)
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
        ax3.scatter(x, data, alpha=0.6, s=10, color='tab:blue', zorder=3)

    # Set labels and formatting
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'$10^{{{int(np.log10(frac))}}}$' if i % 2 == 0 else '' for i, frac in enumerate(train_fracs)])
    ax3.set_xlabel('Nb. of\ntraining samples', fontsize=8)
    # ax3.set_ylabel('RMSE')

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
    ax4 = inset_axes(ax2, width="40%", height="40%", loc='upper right', bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=ax2.transAxes)
    
    # Plot predictions vs observations
    mask = gift_dataset[["sr", "predicted_sr"]].dropna()
    x = mask["sr"]
    y = mask["predicted_sr"]
    
    ax4.scatter(x, y, alpha=0.6, s=10, color='tab:purple')
    
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
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    # ax4.set_title('Model vs GIFT\nobservations')
    
    # Add grid lines to ax1 and ax2
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add panel labels (a, b, c, d) in Nature style
    ax1.text(0.1, 0.1, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    ax2.text(0.1, 0.1, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    ax3.text(0.9, 0.95, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    ax4.text(0.15, 0.95, 'd', transform=ax4.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{Path(__file__).stem}.pdf", dpi=300, bbox_inches='tight')
