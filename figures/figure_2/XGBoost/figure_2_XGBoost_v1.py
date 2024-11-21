# V1 uses EVA_EUNIS_Chelsa_20000_block_relative_extent_0.2 augmented dataset

import numpy as np
import seaborn as sns

import torch.optim as optim

from skorch import NeuralNetRegressor
from skorch.helper import SliceDict
from skorch.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.NNSAR import NNSAR2, SimpleNNBatchNorm
from sklearn.model_selection import cross_validate, GroupKFold

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.plotting import read_result, ResultData, boxplot_bypreds

import pickle

import scipy.stats as stats
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent / Path("../../scripts/XGBoost/")))
import XGBoost_fit_pred_comb_SBCV as XGBoost_fit

PATH_AUGMENTED_DATA = Path(Path(__file__).parent,
    "../../results/EVA_polygons_CHELSA/EVA_EUNIS_CHELSA/EVA_EUNIS_Chelsa_20000_block_relative_extent_0.2.pkl"
)

def process_results(path_results=PATH_AUGMENTED_DATA):
    """
    Reading and processing compiled data.
    """
    results = read_result(path_results)
    gdf_full = results["SAR_data"]
    config = results["config"]
    aggregate_labels = config["env_vars"] + ["std_" + env for env in config["env_vars"]]

    gdf_full = gdf_full.reset_index()
    gdf_nonan = gdf_full.dropna(axis=0, how="any")
    print("we have", len(gdf_full), "entries")
    print("we have", len(gdf_nonan), "entries with non nans")
    print("Filtering rows with nans")
    gdf_full = gdf_nonan

    gdf_full["log_area"] = np.log(gdf_full["area"].astype(np.float64))  # area
    gdf_full["log_sr"] = np.log(gdf_full["sr"].astype(np.float64))

    gdf_full.reset_index(inplace=True, drop=True)
    return ResultData(gdf_full, config, aggregate_labels, None)



# def evaluate_model_all_residuals(gdf, predictors):
#     # fitting model with train test split
#     train_cv_partition_idx, test_cv_partition_idx = train_test_split(
#         gdf.partition.unique(), 
#         test_size=0.2, 
#         random_state=1
#     )
    
#     train_idx = gdf.index[gdf.partition.isin(train_cv_partition_idx)]
#     test_idx = gdf.index[gdf.partition.isin(test_cv_partition_idx)]

#     result_all = {}
    

#     print("Evaluation of model with area")
#     results = {}
#     reg = NeuralNetRegressor(module=SimpleNNBatchNorm,
#                             module__input_dim=1,
#                             **NNSAR_fit.CONFIG["torch_params"])

#     reg.fit(X["log_area"][train_idx], y[train_idx],)
#     results["residuals"] = reg.predict(X["log_area"][test_idx]) - y[test_idx]
#     results["log_area"] =  y[test_idx]
#     result_all["area"] = results
    
#     ###########
#     # climate #
#     ###########
#     print("Evaluation of model with climate")
#     results = {}
#     reg = NeuralNetRegressor(module=SimpleNNBatchNorm,
#                             module__input_dim=X["env_pred"].shape[1],
#                             **NNSAR_fit.CONFIG["torch_params"])
#     reg.fit(X["env_pred"][train_idx], y[train_idx],)
#     results["residuals"] = reg.predict(X["env_pred"][test_idx]) - y[test_idx]
#     results["log_area"] =  y[test_idx]
#     result_all["climate"] = results

#     ##################
#     # area + climate #
#     ##################
#     print("Evaluation of model with area and climate")
#     results = {}
#     reg = NeuralNetRegressor(module=NNSAR2,
#                             module__input_dim=X["env_pred"].shape[1],
#                             **NNSAR_fit.CONFIG["torch_params"])
#     reg.fit(SliceDict(**X)[train_idx], y[train_idx],)
#     results["residuals"] = reg.predict(SliceDict(**X)[test_idx]) - y[test_idx]
#     results["log_area"] =  y[test_idx]
#     result_all["area+climate"] = results
    
#     return result_all


if __name__ == "__main__":
    
    with open(XGBoost_fit.PATH_RESULTS, 'rb') as file:
        result_modelling = pickle.load(file)["result_modelling"]

    for val in result_modelling.values():
        for val2 in val.values():
            val2["test_MSE"] = -val2["test_neg_mean_squared_error"]
        

    PREDICTORS = ["area", "climate", "area+climate, habitat agnostic", "area+climate", "one-hot"]

    # plotting results for test data
    fig = plt.figure(figsize=(6, 6))
    nclasses = len(list(result_modelling.keys()))
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5,1])
    
    # first axis
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_ylim(0.1, 0.8)
    boxplot_bypreds(
        result_modelling,
        ax=ax1,
        spread=0.5,
        colormap="Set2",
        legend=True,
        xlab="",
        ylab="MSE",
        yscale="linear",
        yname="test_MSE",
        habitats=habitats,
        predictors=PREDICTORS,
        widths=0.1,
    )
    label_l1 = ["Forests", "Grasslands", "Mires", "Shrublands"]
    for i,x in enumerate(np.arange(1, len(habitats), step=2)):
        ax1.text(x+0.5, -0., label_l1[i], ha='center', va='bottom', fontsize=10, color='black')
    fig.savefig(Path(__file__).stem + "_model_score.png")

    if False:
        dataset = process_results()
        gdf_full = dataset.gdf
        gdf = gdf_full[gdf_full.habitat_id == "all"].sample(frac=1).reset_index(drop=True)

        predictors = ["log_area"] + dataset.aggregate_labels
        results_residuals = evaluate_model_all_residuals(gdf, predictors)
    
    # second axis
    color_palette = sns.color_palette("Set2", 4)
    qr_range = [0.05, 0.95]
    ax2 = fig.add_subplot(gs[1, 0], )
    x = results_residuals["area"]["log_area"]
    residuals = results_residuals["area"]["residuals"]
    q1_ax2, q3_ax2 = np.quantile(residuals, qr_range)
    ax2.axhspan(q1_ax2, q3_ax2, color=color_palette[0], alpha=0.1)
    ax2.scatter(x, residuals, s=3.0, label="area", color = color_palette[0], alpha=1)
    ax2.set_ylabel("Residuals\n$\\hat{\log \\text{SR}} - \log \\text{SR}$")
    ax2.set_ylim(-2.5,2.5)

    ax3 = fig.add_subplot(gs[1, 1],sharey=ax2)
    x = results_residuals["climate"]["log_area"]
    residuals = results_residuals["climate"]["residuals"]
    q1_ax3, q3_ax3 = np.quantile(residuals, qr_range)
    ax3.axhspan(q1_ax3, q3_ax3, color=color_palette[1], alpha=0.1)
    ax3.scatter(x, residuals, s=3.0, label="climate", color = color_palette[1], alpha=0.8)



    ax4 = fig.add_subplot(gs[1, 2], sharey=ax2,)
    x = results_residuals["area+climate"]["log_area"]
    residuals = results_residuals["area+climate"]["residuals"]
    q1_ax4, q3_ax4 = np.quantile(residuals, qr_range)
    ax4.axhspan(q1_ax4, q3_ax4, color=color_palette[3], alpha=0.1)
    ax4.scatter(x, residuals, s=3.0, label="area + climate", color = color_palette[3], alpha=0.8)
    
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)

    for ax in [ax2,ax3,ax4]:
        ax.plot([min(x), max(x)], [0, 0], c="grey", linestyle="--")
        ax.set_xlabel("log(A)")

    _let = ["A", "B", "C", "D"]
    for i,ax in enumerate([ax1, ax2,ax3,ax4]):
        _x = -0.1
        ax.text(_x, 1.05, _let[i],
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="left",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    fig.savefig("figure_2.png", dpi=300, transparent=True)

    with open("paired_t_test_EVA_EUNIS.txt", "w") as file:
        for j, c in enumerate(habitats[:-1]):
            print(c, file=file)
            performance_baseline = -result_modelling[c]["area"][
                "test_neg_mean_squared_error"
            ]
            performance_new_features = -result_modelling[c]["area+climate"][
                "test_neg_mean_squared_error"
            ]
            t_statistic, p_value = stats.ttest_rel(
                performance_baseline, performance_new_features
            )
            print("T-statistic:", t_statistic, file=file)
            print("P-value:", p_value, file=file)
