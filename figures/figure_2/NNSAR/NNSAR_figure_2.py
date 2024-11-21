import numpy as np
import seaborn as sns

import torch.optim as optim

from skorch import NeuralNetRegressor
from skorch.helper import SliceDict
from skorch.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.NNSAR import NNSAR2, SimpleNNBatchNorm, SAR
from sklearn.model_selection import cross_validate, GroupKFold

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.plotting import read_result, ResultData, boxplot_bypreds

import pickle

import scipy.stats as stats
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent / Path("../../../scripts/NNSAR/")))
import NNSAR_fit_pred_comb_SBCV_plot_megaplot as NNSAR_fit


def evaluate_model_all_residuals(result_fit_split, hab):
    result_all = {}
    climate_predictors  = results_fit_split["climate_predictors"]
    torch_params = NNSAR_fit.CONFIG["torch_params"]

    feature_scaler, target_scaler = result_fit_split[hab]["scalers"]
    gdf = result_fit_split[hab]["gdf"]
    train_idx = result_fit_split[hab]["train_idx"]
    
    X_train, y_train, _, _ = NNSAR_fit.get_Xy_scaled(gdf[train_idx], climate_predictors, feature_scaler=feature_scaler,  target_scaler=target_scaler)
    X_test, y_test, _, _ = NNSAR_fit.get_Xy_scaled(gdf[~train_idx], climate_predictors, feature_scaler=feature_scaler,  target_scaler=target_scaler)
    result_all["log_area"] =  X_test["log_area"]

    ########
    # area #
    ########
    print("Evaluation of model with area")
    reg = NeuralNetRegressor(module=SAR,
                                **torch_params)
    reg.fit(X_train["log_area"], y_train)
    result_all["area"] = reg.predict(X_test["log_area"]) - y_test
    
    ###########
    # climate #
    ###########
    print("Evaluation of model with climate")
    reg = NeuralNetRegressor(module=SimpleNNBatchNorm,
                            module__input_dim=X_train["env_pred"].shape[1],
                            **torch_params)
    reg.fit(X_train["env_pred"], y_train)
    result_all["climate"] = reg.predict(X_test["env_pred"]) - y_test

    ##################
    # area + climate #
    ##################
    print("Evaluation of model with area and climate")
    reg = NeuralNetRegressor(module=NNSAR2,
                            module__input_dim=X_train["env_pred"].shape[1],
                            **torch_params)
    reg.fit(X_train, y_train)
    result_all["area+climate"] = reg.predict(X_test) - y_test
    
    return result_all


if __name__ == "__main__":
    
    with open(NNSAR_fit.PATH_RESULTS, 'rb') as file:
        result_modelling = pickle.load(file)["result_modelling"]

    for hab, val in result_modelling.items():
        print(hab)
        mse_arr = []
        # removing nan values
        for val2 in val.values():
            neg_mse = val2["test_neg_mean_squared_error"]
            if neg_mse.size > 0:
                mse_arr.append(-neg_mse)
        mse = np.stack(mse_arr)
        non_nan_columns = ~np.isnan(mse).any(axis=0)
        matrix_no_nan_columns = mse[:, non_nan_columns]
        for i, val2 in enumerate(val.values()):
            if val2["test_neg_mean_squared_error"].size > 0:
                val2["test_MSE"] = mse[i,:]
            else:
                val2["test_MSE"] = np.array([])
        

    PREDICTORS = ["area", "climate", "area+climate, habitat agnostic", "area+climate"]

    # plotting results for test data
    fig = plt.figure(figsize=(6, 6))
    nclasses = len(list(result_modelling.keys()))
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    # habitats = ["T1", "T3", "S2", "S3"]
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
    

    if True:
        # TOCHANGE
        # result_path = Path(__file__).parent / Path("../../../scripts/NNSAR/NNSAR_fit_simple_plot_megaplot.pkl")
        result_path = Path(__file__).parent / Path("../../../scripts/NNSAR/NNSAR_fit_simple.pkl")

        with open(result_path, 'rb') as file:
            results_fit_split = pickle.load(file)["result_modelling"]

        results_residuals = evaluate_model_all_residuals(results_fit_split, "all")
    
    # second axis
    color_palette = sns.color_palette("Set2", 4)
    qr_range = [0.05, 0.95]
    ax2 = fig.add_subplot(gs[1, 0], )
    x = results_residuals["log_area"]
    residuals = results_residuals["area"]
    q1_ax2, q3_ax2 = np.quantile(residuals, qr_range)
    ax2.axhspan(q1_ax2, q3_ax2, color=color_palette[0], alpha=0.1)
    ax2.scatter(x, residuals, s=3.0, label="area", color = color_palette[0], alpha=1)
    ax2.set_ylabel("Residuals\n$\\hat{\log \\text{SR}} - \log \\text{SR}$")
    ax2.set_ylim(-2.5,2.5)

    ax3 = fig.add_subplot(gs[1, 1],sharey=ax2)
    residuals = results_residuals["climate"]
    q1_ax3, q3_ax3 = np.quantile(residuals, qr_range)
    ax3.axhspan(q1_ax3, q3_ax3, color=color_palette[1], alpha=0.1)
    ax3.scatter(x, residuals, s=3.0, label="climate", color = color_palette[1], alpha=0.8)



    ax4 = fig.add_subplot(gs[1, 2], sharey=ax2,)
    residuals = results_residuals["area+climate"]
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
