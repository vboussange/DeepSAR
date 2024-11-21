"""
This script compares the performance of a XGBoost model and a NN-based model
We do a spatial block cross validation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from skorch import NeuralNetRegressor
from skorch.helper import SliceDict
from skorch.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from src.plotting import COLOR_PALETTE
from src.sar import NNSAR2
from src.utils import save_to_pickle

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / Path("../../figures/figure_2/")))
import NNSAR_figure_2

CONFIG = {
    "torch_params": {
        "optimizer":optim.Adam,
        "lr":5e-3,
        "batch_size":256,
        "max_epochs":300,
        "callbacks":[("early_stopping", EarlyStopping(patience=20))],
        "optimizer__weight_decay":1e-4,
    },
}


def get_Xy_scaled(gdf, predictors, feature_scaler=None, target_scaler=None):
    
    features = gdf[predictors].values.astype(np.float32)
    target = gdf["log_sr"].values.astype(np.float32)
    
    if feature_scaler == None:
        # Standardize features and target
        feature_scaler = StandardScaler()
        feature_scaler.fit(features)

        target_scaler = StandardScaler()
        target_scaler.fit(target.reshape(-1,1))

    features_scaled = feature_scaler.transform(features)
    target_scaled = target_scaler.transform(target.reshape(-1, 1))

    # discarding small areas
    # idx = features_scaled[:, -1] > pd.DataFrame(features_scaled[:, -1]).quantile(0.25)[0]

    # Prepare training data
    X_train = {
        "env_pred": features_scaled[:, 1:],
        "log_area": features_scaled[:, 0].reshape(-1, 1)
    }
    
    return X_train, target_scaled, feature_scaler, target_scaler


def evaluate_nn_mean_val(dataset, predictors, habitats):
    result_all = {}
    gdf_full = dataset.gdf
    
    for hab in habitats:
        print("Training", hab)
        results = {}
        result_all[hab] = results
        
        # here we filter the df and shuffle the rows, so that the folds are randomly selected
        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        gdf = gdf_full[gdf_full.habitat_id == hab].sample(frac=1).reset_index(drop=True)

        train_cv_partition_idx, test_cv_partition_idx = train_test_split(
            gdf.partition.unique(), test_size=0.3, random_state=42
        )
        # train_idx = gdf.index[gdf.partition.isin(train_cv_partition_idx)]
        # test_idx = gdf.index[gdf.partition.isin(test_cv_partition_idx)]
        train_idx = gdf.index
        test_idx = train_idx
        
        X, y, feature_scaler, target_scaler = get_Xy_scaled(gdf, predictors)

        reg = NeuralNetRegressor(module=NNSAR2,
                                module__input_dim=X["env_pred"].shape[1],
                                **CONFIG["torch_params"])
        
        print("Training model...")
        reg.fit(SliceDict(**X)[train_idx], y[train_idx],)
        results["reg"] = reg

        model = reg.module_
        nn = reg.module_.nn
            
        with torch.no_grad():
            sr = model(torch.tensor(X["env_pred"][test_idx],  dtype=torch.float), torch.tensor(X["log_area"][test_idx])).numpy()
            out = nn(torch.tensor(X["env_pred"][test_idx],  dtype=torch.float)).numpy()
            log_c = out[:,0]
            z = out[:, 1]

        results["log_c"] = target_scaler.scale_[0] * (log_c - z * feature_scaler.mean_[0] / feature_scaler.scale_[0]) + target_scaler.mean_[0]
        results["z"] = target_scaler.scale_[0] * z / feature_scaler.scale_[0]
        
        y_transform = results["log_c"] + results["z"] * gdf["log_area"][test_idx].values
        y_true_transform = target_scaler.inverse_transform(reg.predict(SliceDict(**X)[test_idx])).flatten()
        assert np.allclose(y_transform, y_true_transform, atol=1e-5)

    return result_all

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')
    
def plot_params(results, habitats, color_palette):
    data_log_c = [results[hab]['log_c'] for hab in habitats if 'log_c' in results[hab]]
    data_z = [results[hab]['z'] for hab in habitats if 'z' in results[hab]]

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(6, 12))
    
    for i, hab in enumerate(habitats):
        ax = axes.flatten()[i]
        log_c = results[hab]['log_c']
        z = results[hab]['z']
        ax.scatter(log_c, z, c = COLOR_PALETTE[i], alpha=0.5, linewidth=0, s=1)
        ax.set_title(hab)
        ax.set_xlabel('log(c)')
        ax.set_ylabel('z')
        

        
    # # Boxplot for mean_nn1
    # bplot_0 = axes[0].violinplot(data_log_c,showextrema=False)
    # # axes[0].set_title('Mean NN1 by Habitat')
    # axes[0].set_xlabel('Habitat')
    # axes[0].set_ylabel('log(c)')

    # # Boxplot for mean_nn2
    # bplot_1 = axes[1].violinplot(data_z,showextrema=False)
    # # axes[1].set_title('Mean NN2 by Habitat')
    # axes[1].set_xlabel('Habitat')
    # axes[1].set_ylabel('z')
    
    # # fill with colors
    # for ax in axes.flatten():
    #     set_axis_style(ax, habitats)
    # for bplot in [bplot_0, bplot_1]:
    #     for patch, color in zip(bplot['bodies'], color_palette[0:len(bplot['bodies'])]):
    #         patch.set_facecolor(color)
    #         patch.set_alpha(0.9)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
    return fig, axes

if __name__ == "__main__":
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    # habitats = ["T1", "S3"]

    result_path = Path(str(Path(__file__).parent / Path(__file__).stem) + ".pkl")
    
    if True:
        dataset = process_results()
        predictors = ["log_area"] + dataset.aggregate_labels

        results_fit_split = evaluate_nn_mean_val(dataset, predictors, habitats)
        save_to_pickle(result_path ,results_fit_split=results_fit_split)

    else:
        with open(result_path, 'rb') as file:
            results_fit_split = pickle.load(file)["results_fit_split"]
            
    
    # Call the plotting function
    fig, axes = plot_params(results_fit_split, habitats, COLOR_PALETTE)
