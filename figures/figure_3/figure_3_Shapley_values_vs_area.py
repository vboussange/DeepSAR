"""
This script generates plots of Shapley values vs area and residuals vs area for different habitats.

# TODO: change color maps for plotting.CMAP_BR blue and reds
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import geopandas as gpd
from captum.attr import ShapleyValueSampling

sys.path.append(str(Path(__file__).parent / "../../scripts/"))
from src.neural_4pweibull import initialize_ensemble_model
from train import Config, Trainer

# Configuration
MODEL_NAME = "MSEfit_lowlr_nosmallmegaplots2_basearch6_0b85791"
PLOT_CONFIG = [("Area", "tab:green"), ("Mean climate", "tab:blue"), ("Climate heterogeneity", "tab:red")]
N_BINS = 20
SAMPLES_PER_BIN = np.inf

class ShapleyAnalyzer:
    """Handles Shapley value computation and analysis."""
    
    def __init__(self, model, results_fit_split):
        self.model = model
        self.predictors = results_fit_split["predictors"]
        self.feature_scaler = results_fit_split["feature_scaler"]
        
    def _sample_data_by_area(self, gdf, n_bins=100, samples_per_bin=SAMPLES_PER_BIN):
        """Sample data stratified by log area bins."""
        gdf = gdf.copy()
        gdf['log_area_bins'] = pd.cut(gdf['log_megaplot_area'], bins=n_bins, labels=False)
        return gdf.groupby('log_area_bins', group_keys=False).apply(
            lambda x: x.sample(min(samples_per_bin, len(x)))
        )
    
    def compute_shapley_values(self, gdf):
        """Compute Shapley values for given dataframe."""
        # Sample data
        gdf_sampled = self._sample_data_by_area(gdf)
        
        # Prepare features
        X = self.feature_scaler.transform(gdf_sampled[self.predictors].values)
        features = torch.tensor(X, dtype=torch.float32).to(next(self.model.parameters()).device)
        
        # Compute Shapley values
        def forward_fn(X):
            with torch.no_grad():
                # return self.model.models[0].predict_sr(X[:, 1:]).flatten()
                return self.model(X).flatten()

        explainer = ShapleyValueSampling(forward_fn)
        shap_values = explainer.attribute(features).cpu().numpy()
        
        # Create DataFrame
        df_shap = pd.DataFrame(shap_values, columns=self.predictors)
        df_shap["log_megaplot_area_values"] = gdf_sampled["log_megaplot_area"].values
        
        return df_shap, gdf_sampled

class ResidualAnalyzer:
    """Handles residual computation and analysis."""
    
    def __init__(self, model, results_fit_split):
        self.model = model
        self.predictors = results_fit_split["predictors"]
        self.feature_scaler = results_fit_split["feature_scaler"]
        self.target_scaler = results_fit_split["target_scaler"]

    def compute_residuals(self, gdf):
        """Compute model residuals."""
        X = self.feature_scaler.transform(gdf[self.predictors].values)
        features = torch.tensor(X, dtype=torch.float32).to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            predictions = self.model(features).cpu().numpy().flatten()
        
        # Assuming target is species richness
        predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        residuals = gdf["sr"] - predictions
        relative_residuals = residuals / (gdf["sr"])  # Relative residuals
        
        df_residuals = pd.DataFrame({
            'residuals': residuals,
            'megaplot_area': gdf['megaplot_area'].values,
            'log_megaplot_area': gdf['log_megaplot_area'].values,
            'relative_residuals': relative_residuals,
            "coverage": gdf["coverage"].values
        })
        
        return df_residuals

def load_data_and_model():
    """Load model and data."""
    path_results = Path(__file__).parent / f"../../scripts/results/train/checkpoint_{MODEL_NAME}.pth"
    results_fit_split = torch.load(path_results, map_location="cpu")
    config = results_fit_split["config"]
    
    # Load and prepare data
    eva_dataset = gpd.read_parquet(config.path_eva_data)
    eva_dataset["log_megaplot_area"] = np.log(eva_dataset["megaplot_area"])
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    eva_dataset["coverage"] = eva_dataset["log_observed_area"] / eva_dataset["log_megaplot_area"]
    eva_dataset = eva_dataset[eva_dataset["num_plots"] > 10] # todo: to change
    
    # Filter test data
    test_data = eva_dataset[eva_dataset["test"]].copy()
    
    # Initialize model
    model = initialize_ensemble_model(
        results_fit_split["ensemble_model_state_dict"], 
        results_fit_split["predictors"], 
        config
    )
    
    return model, results_fit_split, test_data, config

def aggregate_shapley_features(df_shap, config):
    """Aggregate Shapley values by feature groups."""
    std_features = [f"std_{var}" for var in config.climate_variables]
    mean_features = config.climate_variables
    
    df_shap["Climate heterogeneity"] = np.abs(df_shap[std_features]).sum(axis=1)
    df_shap["Mean climate"] = np.abs(df_shap[mean_features]).sum(axis=1)
    df_shap["Area"] = np.abs(df_shap[["log_megaplot_area"]]).sum(axis=1)
    
    # Normalize to relative values
    feature_cols = ["Area", "Climate heterogeneity", "Mean climate"]
    total_importance = df_shap[feature_cols].sum(axis=1)
    df_shap[feature_cols] = df_shap[feature_cols].div(total_importance, axis=0)
    
    return df_shap

def plot_shapley_values(df_shap, ax, config_plot):
    """Plot Shapley values vs area."""
    for var_name, color in config_plot:
        # Bin by area
        df_shap['area_bins'] = pd.cut(df_shap['log_megaplot_area_values'], bins=N_BINS, labels=False)
        
        # Aggregate by bins
        grouped = df_shap.groupby('area_bins')
        mean_vals = grouped[var_name].mean()
        std_vals = grouped[var_name].std()
        mean_areas = grouped['log_megaplot_area_values'].mean()
        
        # Plot with error bars
        ax.errorbar(np.exp(mean_areas), mean_vals, yerr=std_vals, 
                   fmt='o', color=color, label=var_name, alpha=0.7)
    
    ax.set_xscale("log")
    ax.set_ylabel("Relative absolute Shapley values")

def plot_residuals(df_residuals, ax):
    """Plot relative residuals vs area as scatter plot."""
    # Calculate relative residuals (residuals as percentage of observed values)
    # Assuming 'sr' is available in the original data, we need to add it to df_residuals
    # For now, we'll compute relative residuals as residuals / (residuals + predictions)
    # This requires getting predictions first
    
    # Create scatter plot of relative residuals
    relative_residuals = df_residuals['relative_residuals'] 
    
    # scatter = ax.scatter(df_residuals['megaplot_area'], relative_residuals, 
    #                     c=df_residuals['coverage'], alpha=0.5, s=10, cmap='viridis')
    # plt.colorbar(scatter, ax=ax, label='Coverage')
    # Create bins for area to compute variance trends
    df_residuals['area_bins'] = pd.cut(df_residuals['log_megaplot_area'], bins=5, labels=False)
    
    # Compute statistics for each bin
    grouped = df_residuals.groupby('area_bins')
    mean_residuals = grouped['relative_residuals'].mean()
    std_residuals = grouped['relative_residuals'].std()
    mean_areas = grouped['megaplot_area'].mean()
    
    # Plot the trend with shaded region for ±std
    ax.fill_between(mean_areas, mean_residuals - std_residuals, mean_residuals + std_residuals,
                   alpha=0.3, color='lightgray', label='±1 std')
    
    # Plot individual residuals
    ax.scatter(
        df_residuals['megaplot_area'], relative_residuals, 
        c="tab:orange", alpha=0.5, s=10
    )
    
    # Plot mean trend line
    # ax.plot(mean_areas, mean_residuals, color='red', linewidth=2, label='Mean residuals')
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(relative_residuals.quantile(0.01), -relative_residuals.quantile(0.01))
    
    ax.set_xscale("log")
    ax.set_ylabel("Relative residuals")

if __name__ == "__main__":
    model, results_fit_split, test_data, config = load_data_and_model()
    
    # Initialize analyzers
    shapley_analyzer = ShapleyAnalyzer(model, results_fit_split)
    residual_analyzer = ResidualAnalyzer(model, results_fit_split)
    
    # Compute Shapley values
    df_shap, gdf_sampled = shapley_analyzer.compute_shapley_values(test_data)
    df_shap = aggregate_shapley_features(df_shap, config)
    
    # Compute residuals
    df_residuals = residual_analyzer.compute_residuals(test_data)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # Plot Shapley values
    plot_shapley_values(df_shap, ax1, PLOT_CONFIG)
    
    # Plot residuals
    plot_residuals(df_residuals, ax2)
    
    # Add legend
    ax1.legend(frameon=True, fancybox=True, loc='center left')
    fig.supxlabel("Area (m²)")
    plt.tight_layout()
    
    # Save figure
    fig.savefig(f"{Path(__file__).stem}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

