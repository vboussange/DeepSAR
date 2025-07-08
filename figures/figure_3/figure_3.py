"""
This script generates plots of Shapley values vs area for different habitats.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
from captum.attr import ShapleyValueSampling

from src.neural_4pweibull import initialize_ensemble_model
import sys
sys.path.append(str(Path(__file__).parent / "../../scripts/"))
from src.neural_4pweibull import initialize_ensemble_model
from train import Config, Trainer

# Configuration
MODEL_NAME = "MSEfit_lowlr_nosmallmegaplots2_basearch6_0b85791"
PLOT_CONFIG = [("Area", "#f72585"), ("Climate heterogeneity", "#4cc9f0"), ("Mean climate", "#3a0ca3")]

def sample_data_by_area(gdf, n_bins=100, samples_per_bin=np.inf):
    """Sample data stratified by log area bins."""
    gdf = gdf.copy()
    gdf['log_area_bins'] = pd.cut(gdf['log_megaplot_area'], bins=n_bins, labels=False)
    return gdf.groupby('log_area_bins', group_keys=False).apply(
        lambda x: x.sample(min(samples_per_bin, len(x)))
    )

class ShapleyAnalyzer:
    """Handles Shapley value computation and analysis."""
    
    def __init__(self, model, results_fit_split):
        self.model = model
        self.predictors = results_fit_split["predictors"]
        self.feature_scaler = results_fit_split["feature_scaler"]
    
    def compute_shapley_values(self, gdf):
        """Compute Shapley values for given dataframe."""
        gdf_sampled = sample_data_by_area(gdf)
        X = self.feature_scaler.transform(gdf_sampled[self.predictors].values)
        features = torch.tensor(X, dtype=torch.float32).to(next(self.model.parameters()).device)
        
        def forward_fn(X):
            with torch.no_grad():
                return self.model(X).flatten()

        explainer = ShapleyValueSampling(forward_fn)
        shap_values = explainer.attribute(features, n_samples=400).cpu().numpy()
        
        df_shap = pd.DataFrame(shap_values, columns=self.predictors)
        df_shap["log_megaplot_area_values"] = gdf_sampled["log_megaplot_area"].values
        
        return df_shap

def load_data_and_model():
    """Load model and data."""
    path_results = Path(__file__).parent / f"../../scripts/results/train/checkpoint_{MODEL_NAME}.pth"
    results_fit_split = torch.load(path_results, map_location="cpu")
    config = results_fit_split["config"]
    
    eva_dataset = gpd.read_parquet(config.path_eva_data)
    eva_dataset["log_megaplot_area"] = np.log(eva_dataset["megaplot_area"])
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    
    test_data = eva_dataset[eva_dataset["test"]]
    
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
    
    df_shap["Environmental heterogeneity"] = np.abs(df_shap[std_features]).sum(axis=1)
    df_shap["Mean environmental conditions"] = np.abs(df_shap[mean_features]).sum(axis=1)
    df_shap["Area"] = np.abs(df_shap[["log_megaplot_area"]]).sum(axis=1)
    
    feature_cols = ["Area", "Environmental heterogeneity", "Mean environmental conditions"]
    total_importance = df_shap[feature_cols].sum(axis=1)
    df_shap[feature_cols] = df_shap[feature_cols].div(total_importance, axis=0)
    
    return df_shap

def plot_shapley_values(df_shap, ax, config_plot):
    """Plot Shapley values vs area."""
    for var_name, color in config_plot:
        df_shap['area_bins'] = pd.cut(df_shap['log_megaplot_area_values'], bins=20, labels=False)
        grouped = df_shap.groupby('area_bins')
        mean_vals = grouped[var_name].mean()
        std_vals = grouped[var_name].std()
        mean_areas = np.exp(grouped['log_megaplot_area_values'].mean()) / 1e6 
        
        ax.plot(mean_areas, mean_vals, 'o-', color=color, label=var_name, alpha=0.7)
        ci_lower = mean_vals - std_vals 
        ci_upper = mean_vals + std_vals 
        ax.fill_between(mean_areas, ci_lower, ci_upper, alpha=0.2, color=color)    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Relative absolute\nShapley values")

if __name__ == "__main__":
    np.random.seed(42)

    model, results_fit_split, test_data, config = load_data_and_model()
    shapley_analyzer = ShapleyAnalyzer(model, results_fit_split)
    df_shap = shapley_analyzer.compute_shapley_values(test_data)
    df_shap = aggregate_shapley_features(df_shap, config)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_shapley_values(df_shap, ax, PLOT_CONFIG)
    
    ax.legend(frameon=True, fancybox=True, bbox_to_anchor=(0.5, 1.2), loc='center')
    ax.set_ylim(1e-2, 1.5)
    fig.supxlabel("Area (km²)")
    fig.tight_layout()
    ax.grid(True, alpha=0.3)
    fig.savefig("figure_3.pdf", dpi=300, bbox_inches='tight')
    plt.show()
