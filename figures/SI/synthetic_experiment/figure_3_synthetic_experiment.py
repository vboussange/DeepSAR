""""
Plotting figure 3 'Feature importance'
"""

import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from src.model_validation import get_spatial_block_cv_index
from xgboost import XGBRegressor

from src.data_processing.utils_env_pred import calculate_aggregates, CHELSADataset
from src.data_processing.utils_landcover import CopernicusDataset
from src.data_processing.utils_polygons import partition_polygon_gdf
import shap
import sys
from figure_2_synthetic_experiment import (
    process_results,
    CONFIG,
)

def evaluate_model_per_hab(res, xgb_params, habitats, predictors):
    result_all = {}
    gdf_full = res.gdf
    train_cv_partition_idx, test_cv_partition_idx = train_test_split(gdf_full.partition.unique(), test_size=0.3, random_state=42)
    for hab in habitats:
        gdf = gdf_full[gdf_full.habitat_id == hab].reset_index(drop=True)
        gdf_train = gdf[gdf.partition.isin(train_cv_partition_idx)]
        X_train = gdf_train[predictors]  # env vars
        y_train = gdf_train["log_sr"]
        reg = XGBRegressor(**xgb_params)
        reg.fit(X_train, y_train)
        explainer = shap.Explainer(reg)
        gdf_test = gdf[gdf.partition.isin(test_cv_partition_idx)]
        X_test = gdf_test[predictors]  # env vars
        result_all[hab] = explainer(X_test)
    return result_all

def get_df_shap_val(dataset, result_modelling):
    shap_vals = []
    for hab in dataset.config["habitats"]:
        shap_res = result_modelling[hab]
        # Calculate the mean absolute Shapley values across all classes and samples
        shap_abs = shap_res.values # Absolute values for each class
        df_shap_values = pd.DataFrame(np.stack(shap_abs), columns=result_modelling[dataset.config["habitats"][0]].feature_names) # Mean across all samples and classes
        df_shap_values["habitat_names"] = hab

        shap_vals.append(df_shap_values)

    df_shap_vals = pd.concat(shap_vals, axis=0)
    return df_shap_vals

def boxplot_byclass(df=None,
                    ax=None,
                    spread=None,
                    colormap=None,
                    legend=False,
                    xlab=None,
                    ylab=None,
                    habitats = None,
                    predictors = None,
                    widths=0.1):
    if not habitats:
        habitats = list(df.habitat_names.unique())
    N = len(habitats)  # number of habitats
    color_palette = sns.color_palette(colormap, N)
    M = len(predictors)  # number of groups
    
    dfg = df.groupby("habitat_names")
    for j, hab in enumerate(habitats):
        dfhab = dfg.get_group(hab)
        y = [
            dfhab[k] for k in predictors
        ]
        xx = np.arange(1, M + 1) + (
            j - N/ 2
        ) * spread / N  # artificially shift the x values to better visualise the std
        violin_parts = ax.violinplot(y, positions=xx, widths=widths, showextrema=False, vert=False)
        for pc in violin_parts['bodies']:
            pc.set_facecolor(color_palette[j])
            pc.set_edgecolor(color_palette[j])
            pc.set_alpha(1)
        
    ax.set_xlabel(ylab)
    ax.set_ylabel(xlab)
    x = predictors
    ax.set_yticks(np.arange(1, len(x) + 1) - 0.5)
    ax.set_yticklabels(x)
    if legend:
        ax.legend(handles=[
            Line2D([0], [0], color=color_palette[i], label=habitats[i])
            for i in range(len(habitats))
        ])
        plt.show()


if __name__ == "__main__":
    dataset = process_results()
    dataset.config["habitats"] = ["DBF_closed"]
    dataset.gdf["habitat_id"] = "DBF_closed"
    predictors= ["log_area"]+dataset.aggregate_labels
    result_modelling = evaluate_model_per_hab(dataset, CONFIG["xgb_params"], dataset.config["habitats"],predictors)
    
    df_shap_vals = get_df_shap_val(dataset, result_modelling)
    
    df_shap_importance = df_shap_vals[dataset.aggregate_labels].abs().mean(axis=0).sort_values(ascending=False)
    name_most_important_features = df_shap_importance.index[:5].tolist()

    nclasses = len(list(result_modelling.keys()))
    print(result_modelling.keys())
    color_palette = sns.color_palette("Set2", len(dataset.config["habitats"]))

    fig, axs = plt.subplots(1,2, figsize=(15,4))

    ax = axs[1]
    # plotting area effect
    for i, k in enumerate(dataset.config["habitats"]):
        ax.scatter(result_modelling[k][:, "log_area"].data,
                   result_modelling[k][:, "log_area"].values,
                   color=color_palette[i],
                   label=k,
                   alpha=0.5,
                   s=0.5)
    ax.set_ylabel("Shapley value for\nlog_area")
    ax.set_xlabel("log_area")
    
    boxplot_byclass(df=df_shap_vals, 
                ax=axs[0], 
                spread=.8,
                colormap="Set2",
                legend=True,
                xlab="Predictors",
                ylab="Shapley values",
                habitats=dataset.config["habitats"],
                predictors=name_most_important_features,
                widths=0.4)
    
    # plotting violin plot of climate variables per cohort
    
    # plotting climate var 2 effect

    fig.savefig("figure_3_synthetic.png", dpi=300, transparent=True)







