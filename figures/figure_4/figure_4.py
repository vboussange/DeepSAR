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

from muscari import MuScaRiEnsemble

ROOT = Path(__file__).parents[2]
RUN_DIR = ROOT / "scripts" / "results" / "train" / "ceacce0"

# Configuration
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
PLOT_CONFIG = [
    ("Area", r"Spatial unit area, $A$", "#f72585", "o", "-"),
    ("Environmental heterogeneity", "Environmental heterogeneity", "#4cc9f0", "s", "-"),
    ("Mean environmental conditions", "Mean environmental conditions", "#3a0ca3", "^", "-"),
]

class ShapleyAnalyzer:
    """Handles Shapley value computation and analysis."""
    
    def __init__(self, model):
        model.eval()
        self.model = model.models[0]  # use the first model of the ensemble
    
    def compute_shapley_values(self, gdf):
        """Compute Shapley values for given dataframe."""        
        features = torch.tensor(
            gdf[["log_observed_area"] + self.model.feature_names].values,
            dtype=torch.float32,
        )
        feature_scaler = self.model.feature_scaler
        X = torch.tensor(feature_scaler.transform(features), dtype=torch.float32).to(next(self.model.parameters()).device)
        X = X[:, 1:]  # Exclude the first column (log_observed_area)
        
        def forward_fn(X):
            with torch.no_grad():
                return self.model._predict_sr_tot(X).flatten()

        explainer = ShapleyValueSampling(forward_fn)
        shap_values = explainer.attribute(X, n_samples=100).cpu().numpy()
        
        df_shap = pd.DataFrame(shap_values, columns=self.model.feature_names)
        df_shap["log_sp_unit_area_values"] = gdf["log_sp_unit_area"].values
        
        return df_shap

def load_data_and_model():
    """Load model and data."""
    model, config = MuScaRiEnsemble.from_folds(RUN_DIR, device=DEVICE, return_config=True)
    eva_dataset = gpd.read_parquet(config.path_sbcv_data / "fold_0_test.parquet")
    eva_dataset["log_sp_unit_area"] = np.log(eva_dataset["sp_unit_area"])
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    return model, config, eva_dataset

def aggregate_shapley_features(df_shap):
    """Aggregate Shapley values by feature groups."""
    feature_names = df_shap.columns.tolist()
    std_features = [f for f in feature_names if f.startswith("std_")]
    mean_features = [
        f
        for f in feature_names
        if f not in std_features
        and f != "log_sp_unit_area"
        and f != "log_sp_unit_area_values"
    ]

    df_shap["Environmental heterogeneity"] = (
        np.abs(df_shap[std_features]).sum(axis=1) if std_features else 0.0
    )
    df_shap["Mean environmental conditions"] = (
        np.abs(df_shap[mean_features]).sum(axis=1) if mean_features else 0.0
    )
    df_shap["Area"] = (
        np.abs(df_shap[["log_sp_unit_area"]]).sum(axis=1)
        if "log_sp_unit_area" in df_shap.columns
        else 0.0
    )

    feature_cols = [
        "Area",
        "Environmental heterogeneity",
        "Mean environmental conditions",
    ]
    total_importance = df_shap[feature_cols].sum(axis=1)
    df_shap[feature_cols] = df_shap[feature_cols].div(total_importance, axis=0)
    
    return df_shap

def plot_shapley_values(df_shap, ax, config_plot):
    """Plot Shapley values vs area."""
    for var_name, label, color, marker, linestyle in config_plot:
        df_shap['area_bins'] = pd.cut(df_shap['log_sp_unit_area_values'], bins=20, labels=False)
        grouped = df_shap.groupby('area_bins')
        mean_vals = grouped[var_name].mean()
        std_vals = grouped[var_name].std()
        mean_areas = np.exp(grouped['log_sp_unit_area_values'].mean()) / 1e6 
        
        ax.plot(
            mean_areas,
            mean_vals,
            marker=marker,
            markersize=4,
            linestyle=linestyle,
            color=color,
            label=label,
            alpha=0.8,
        )
        ci_lower = mean_vals - std_vals 
        ci_upper = mean_vals + std_vals 
        ax.fill_between(mean_areas, ci_lower, ci_upper, alpha=0.2, color=color)    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Normalized absolute\nShapley values")

if __name__ == "__main__":
    np.random.seed(42)

    model, config, test_data = load_data_and_model()
    shapley_analyzer = ShapleyAnalyzer(model)
    df_shap = shapley_analyzer.compute_shapley_values(test_data)
    df_shap = aggregate_shapley_features(df_shap)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_shapley_values(df_shap, ax, PLOT_CONFIG)
    
    ax.legend(frameon=True, fancybox=True, bbox_to_anchor=(0.5, 1.2), loc='center')
    # ax.set_ylim(1e-2, 1.5)
    fig.supxlabel(r"Spatial unit area, $A$ (km²)")
    fig.tight_layout()
    ax.grid(True, alpha=0.3)
    fig.savefig("figure_4.pdf", dpi=300, bbox_inches='tight')
    plt.show()
