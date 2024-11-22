""""
Plotting figure 2 'prediction power of climate, area, and both on SR'
# TODO: fix the test indices which currently do not work
"""

import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from sklearn.metrics import mean_squared_error

from src.plotting import boxplot_bypreds

import scipy.stats as stats

import sys
sys.path.append(str(Path(__file__).parent / Path("../../../scripts/MLP3/")))
from MLP_fit_torch_SBCV import Config
import MLP_fit_torch_SBCV as MLP_fit
from src.mlp import MLP

def load_model_checkpoint(model_state, predictors):
        """Load the model and scalers from the saved checkpoint."""
        
        model = MLP(len(predictors))
        model.load_state_dict(model_state)
        model.eval()
        return model
    
def evaluate_residuals(gdf, checkpoint, fold):
    predictors = checkpoint["predictors"]
    # TODO: to be fixed
    test_idx = np.random.choice(checkpoint["test_idx"][fold], 1000, replace=False)
    # test_idx = gdf.sample(1000).index
    feature_scaler, target_scaler = checkpoint["feature_target_scaler"][fold]
    gdf_test = gdf.loc[test_idx,:]
    X_test, y_test, _, _ = MLP_fit.scale_features_targets(gdf_test, predictors, feature_scaler=feature_scaler,  target_scaler=target_scaler)
    log_area =  gdf_test["log_area"].values
    
    model_state = checkpoint["model_state_dict"][fold]
    model = load_model_checkpoint(model_state, predictors)
    with torch.no_grad():
        y = target_scaler.inverse_transform(model(X_test))
        y_test = target_scaler.inverse_transform(y_test)
        res = y - y_test
        print(f"MSE: {mean_squared_error(y, y_test):.4f}")
    return log_area, res.flatten()

def evaluate_model_all_residuals(gdf, result_modelling, hab):
    fold = 5

    result_all = {}

    for scenario in ["area", "climate", "area+climate"]:
        checkpoint = result_modelling[hab][scenario]
        log_area, residuals = evaluate_residuals(gdf, checkpoint, fold)
        result_all[scenario] = {}
        result_all[scenario]["residuals"] = residuals
        result_all[scenario]["log_area"] = log_area
    return result_all


if __name__ == "__main__":
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    MODEL = "very_small"
    seed = 1
    path_results = f"../../../scripts/MLP3/results/MLP_fit_torch_SBCV_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint.pth"
    
    result_modelling = torch.load(path_results, map_location="cpu")
        
    # path_results_no_physics = "../../../scripts/MLP3/MLP_fit_torch_SBCVdSRdA_weight_0e+00.pkl"
    # with open(path_results_no_physics, 'rb') as file:
    #     result_modelling_no_physics = pickle.load(file)["result_modelling"]
        
    for hab in habitats:
        val = result_modelling[hab]
        mse_arr = []
        # removing nan values
        for val2 in val.values():
            mse = val2["test_MSE"]
            if len(mse) > 0:
                mse_arr.append(mse)
        mse = np.stack(mse_arr)
        non_nan_columns = ~np.isnan(mse).any(axis=0)
        matrix_no_nan_columns = mse[:, non_nan_columns]
        for i, val2 in enumerate(val.values()):
            if len(val2["test_MSE"]) > 0:
                val2["test_MSE"] = mse[i,:]
            else:
                val2["test_MSE"] = np.array([])
                
    # Calculating residuals
    hab = "all"
    # dataset = MLP_fit.process_results()
    # if not "config" in result_modelling.keys():
    #     result_modelling["config"] = {"seed": 2}
    config = result_modelling["config"]
    gdf = MLP_fit.load_preprocessed_data(hab, config.hash_data, config.data_seed)
    results_residuals = evaluate_model_all_residuals(gdf, result_modelling, hab)
    
        
    result_modelling["all"]['area+climate, habitat agnostic'] = {"test_MSE":[]}
    PREDICTORS = ["power_law", 
                  "area", 
                  "climate", 
                  'area+climate, habitat agnostic', 
                  "area+climate", 
                  "area+climate, no physics"
                  ]

    # plotting results for test data
    fig = plt.figure(figsize=(6, 6))
    nclasses = len(list(result_modelling.keys()))
    # habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    # habitats = ["T1", "T3"]
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5,1])
    
    # first axis
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_ylim(0., 0.8)
    boxplot_bypreds(
        result_modelling,
        ax=ax1,
        spread=0.6,
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
    fig.savefig(Path(__file__).stem + "_model_score.png", transparent=True, dpi=300)
    
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

    ax3 = fig.add_subplot(gs[1, 1],sharey=ax2, sharex=ax2)
    x = results_residuals["climate"]["log_area"]
    residuals = results_residuals["climate"]["residuals"]
    q1_ax3, q3_ax3 = np.quantile(residuals, qr_range)
    ax3.axhspan(q1_ax3, q3_ax3, color=color_palette[1], alpha=0.1)
    ax3.scatter(x, residuals, s=3.0, label="climate", color = color_palette[1], alpha=0.8)

    ax4 = fig.add_subplot(gs[1, 2], sharey=ax2, sharex=ax2)
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
            performance_baseline = result_modelling[c]["area"][
                "test_MSE"
            ]
            performance_new_features = result_modelling[c]["area+climate"][
                "test_MSE"
            ]
            t_statistic, p_value = stats.ttest_rel(
                performance_baseline, performance_new_features
            )
            print("T-statistic:", t_statistic, file=file)
            print("P-value:", p_value, file=file)
