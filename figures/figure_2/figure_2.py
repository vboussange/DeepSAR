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
from deepsar.plotting import boxplot_bypreds
import pandas as pd
from matplotlib.patches import Patch
import geopandas as gpd

import sys
sys.path.append(str(Path(__file__).parent / "../../scripts/"))
from deepsar.neural_4pweibull import initialize_ensemble_model
from train import Config, Trainer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import MultiComparison
from deepsar.cld import create_comp_matrix_allpair_t_test, multcomp_letters

import scipy.stats as stats


def report_model_performance_and_bias(df_plot, eva_test_data, gift_dataset, metric, output_file="model_performance_and_bias_report.txt"):
    """
    Report model performance, statistical significance, and relative bias for eva and gift datasets.
    
    Parameters:
    -----------
    df_plot : pd.DataFrame
        Combined dataframe with model results
    eva_test_data : pd.DataFrame
        EVA test dataset with observed and predicted SR values
    gift_dataset : pd.DataFrame
        GIFT dataset with observed and predicted SR values
    metric : str
        Performance metric to analyze
    output_file : str
        Path to output text file for results
    """
    datasets = ['eva', 'gift']
    
    with open(output_file, "w") as file:
        print("Relative bias calculated as (observed - predicted) / observed", file=file)
        print("Positive values indicate model underestimation, negative values indicate overestimation\n", file=file)
        
        # Relative Bias Analysis
        print("RELATIVE BIAS ANALYSIS", file=file)
        print("=" * 50, file=file)
        
        # EVA dataset bias
        eva_mask = eva_test_data[["sr", "predicted_sr"]].dropna()
        eva_observed = eva_mask["sr"]
        eva_predicted = eva_mask["predicted_sr"]
        eva_relative_bias = (eva_observed - eva_predicted) / eva_observed
        
        print("EVA Dataset", file=file)
        print("-" * 20, file=file)
        print(f"Mean relative bias: {eva_relative_bias.mean():.4f}", file=file)
        print(f"Median relative bias: {eva_relative_bias.median():.4f}", file=file)
        print(f"Std relative bias: {eva_relative_bias.std():.4f}", file=file)
        print(f"Min relative bias: {eva_relative_bias.min():.4f}", file=file)
        print(f"Max relative bias: {eva_relative_bias.max():.4f}", file=file)
        print(f"N observations: {len(eva_relative_bias)}", file=file)
        
        # GIFT dataset bias
        gift_mask = gift_dataset[["sr", "predicted_sr"]].dropna()
        gift_observed = gift_mask["sr"]
        gift_predicted = gift_mask["predicted_sr"]
        gift_relative_bias = (gift_observed - gift_predicted) / gift_observed
        
        print("\nGIFT Dataset", file=file)
        print("-" * 20, file=file)
        print(f"Mean relative bias: {gift_relative_bias.mean():.4f}", file=file)
        print(f"Median relative bias: {gift_relative_bias.median():.4f}", file=file)
        print(f"Std relative bias: {gift_relative_bias.std():.4f}", file=file)
        print(f"Min relative bias: {gift_relative_bias.min():.4f}", file=file)
        print(f"Max relative bias: {gift_relative_bias.max():.4f}", file=file)
        print(f"N observations: {len(gift_relative_bias)}", file=file)
        
        # Model Performance and Statistical Significance Analysis
        print("\n\nMODEL PERFORMANCE AND STATISTICAL SIGNIFICANCE", file=file)
        print("=" * 60, file=file)
        
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
                
            print(f"\n{dataset.upper()} Dataset", file=file)
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

    print(f"Model performance and bias analysis saved to '{output_file}'")

if __name__ == "__main__":

    path_neural_weibull_results = Path(f"../../scripts/results/benchmark/neural4p_weibull_nosmallsp_units2_basearch6_0b85791_benchmark.csv")    
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
   
    datasets = ['eva', 'gift']
    colors = ["#f72585","#4cc9f0"]
    
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

    # Third plot: observed vs predicted for area+environment model on EVA dataset
    ax3 = inset_axes(ax1, width="40%", height="40%", loc='upper right', bbox_to_anchor=(-0.05, 0, 1, 1), bbox_transform=ax1.transAxes)
    MODEL_NAME = "MSEfit_lowlr_nosmallsp_units2_basearch6_0b85791"
    # Load model and data for EVA predictions
    eva_data_dir = Path("../../data/processed/EVA/6c2d61d/")
    path_results = Path(f"../../scripts/results/train/checkpoint_{MODEL_NAME}.pth")
    
    # Load model results
    result_modelling = torch.load(path_results, map_location="cpu")
    config = result_modelling["config"]

    # Load EVA dataset
    # Load and prepare data
    eva_dataset = gpd.read_parquet(config.path_eva_data)
    eva_dataset["log_sp_unit_area"] = np.log(eva_dataset["sp_unit_area"])
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    # eva_dataset["coverage"] = eva_dataset["log_observed_area"] / eva_dataset["log_sp_unit_area"]
    # eva_dataset = eva_dataset[eva_dataset["num_plots"] > 10] # todo: to change
    
    # Filter test data
    eva_test_data = eva_dataset[eva_dataset["test"]].sample(n=400)
    
    # Extract model components
    predictors = result_modelling["predictors"]
    feature_scaler = result_modelling["feature_scaler"]
    target_scaler = result_modelling["target_scaler"]
    
    # Make predictions
    X = eva_test_data[predictors].copy()
    X = torch.tensor(feature_scaler.transform(X), dtype=torch.float32)
    
    # Initialize model
    model = initialize_ensemble_model(result_modelling["ensemble_model_state_dict"], predictors, config, "cpu")
    
    with torch.no_grad():
        y_pred = model(X).numpy()
        y_pred = target_scaler.inverse_transform(y_pred)
    
    eva_test_data["predicted_sr"] = y_pred.squeeze()
    
    # Plot predictions vs observations for EVA
    mask_eva = eva_test_data[["sr", "predicted_sr"]].dropna()
    x_eva = mask_eva["sr"]
    y_eva = mask_eva["predicted_sr"]
    eva_relative_bias = (x_eva - y_eva) / x_eva
    eva_median_bias = eva_relative_bias.median()
    ax3.text(0.1, 0.06, f'Rel. bias: {eva_median_bias:.3f}', 
            transform=ax3.transAxes, 
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    ax3.scatter(x_eva, y_eva, alpha=0.6, s=10, color="#f72585")
    
    # Add 1:1 line
    max_val = np.nanmax([x_eva.max(), y_eva.max()])
    min_val = np.nanmin([x_eva.min(), y_eva.min()])
    ax3.plot([min_val, max_val], [min_val, max_val], linestyle='--',color="black", linewidth=1)
    
    ax3.set_xlabel('EVA observed SR', fontsize=8)
    ax3.set_ylabel('Predicted SR', fontsize=8)
    ax3.set_yscale('log')
    ax3.set_xscale('log')

    # Fourth plot: model predictions vs GIFT observations
    # Load model and data for prediction comparison
    gift_data_dir = Path("../../data/processed/GIFT_CHELSA_compilation/6c2d61d/")
    
    # Load GIFT dataset
    gift_dataset = gpd.read_parquet(gift_data_dir / "sp_unit_data.parquet")
    gift_dataset["log_sp_unit_area"] = np.log(gift_dataset["sp_unit_area"])
    gift_dataset["log_observed_area"] = np.log(gift_dataset["sp_unit_area"])
    gift_dataset = gift_dataset.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    
    # Make predictions for GIFT
    X_gift = gift_dataset[predictors].copy()
    X_gift = torch.tensor(feature_scaler.transform(X_gift), dtype=torch.float32)
    
    with torch.no_grad():
        y_pred_gift = model(X_gift).numpy()
        y_pred_gift = target_scaler.inverse_transform(y_pred_gift)
    
    gift_dataset["predicted_sr"] = y_pred_gift.squeeze()
    
    # Create inset axes in ax2
    ax4 = inset_axes(ax2, width="40%", height="40%", loc='upper right', bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=ax2.transAxes)
    
    # Plot predictions vs observations for GIFT
    mask_gift = gift_dataset[["sr", "predicted_sr"]].dropna()
    x_gift = mask_gift["sr"]
    y_gift = mask_gift["predicted_sr"]
    gift_relative_bias = (x_gift - y_gift) / x_gift
    gift_median_bias = gift_relative_bias.median()

    ax4.text(0.1, 0.06, f'Rel. bias: {gift_median_bias:.3f}', 
            transform=ax4.transAxes, 
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    ax4.scatter(x_gift, y_gift, alpha=0.6, s=10, color="#4cc9f0")
    
    # Add 1:1 line
    max_val_gift = np.nanmax([x_gift.max(), y_gift.max()])
    min_val_gift = np.nanmin([x_gift.min(), y_gift.min()])
    ax4.plot([min_val_gift, max_val_gift], [min_val_gift, max_val_gift],  linestyle='--', color="black", linewidth=1)
    
    ax4.set_xlabel('GIFT observed SR', fontsize=8)
    ax4.set_ylabel('Predicted SR', fontsize=8)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    # Add grid lines to ax1 and ax2
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add panel labels (a, b, c, d) in Nature style
    ax1.text(0.1, 0.1, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    ax2.text(0.1, 0.1, 'c', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    ax3.text(0.15, 0.95, 'b', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    ax4.text(0.15, 0.95, 'd', transform=ax4.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{Path(__file__).stem}.pdf", dpi=300, bbox_inches='tight')
    
    report_model_performance_and_bias(df_plot, eva_test_data, gift_dataset, metric)