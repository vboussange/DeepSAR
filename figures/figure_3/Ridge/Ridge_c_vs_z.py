"""
Calculating c and z coeff from Ridge model, plotting.

# TODO
Fix c and z calculation. The only thing that works as of now is:

np.dot(X.values, coef) + intercept == model.predict(F)

Problem is the scaling! Because we scale independently the loga x pred_env_i

Two solutions:
* We use LinearRegression
* We rescale the model parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / Path("../../figures/figure_2/")))
import figure_2_Ridge

CONFIG = {
    "ridge": {"alpha": 0.}
}


def get_Xy(gdf, climate_predictors, feature_scaler=None, target_scaler=None):
    gdf.reset_index(drop=True, inplace=True)
    
    
    df_features = gdf[["log_area"] + climate_predictors]
    if feature_scaler is None:
        feature_scaler = StandardScaler()
        feature_scaler.fit(df_features)
                
    X_features = pd.DataFrame(feature_scaler.transform(df_features), columns=df_features.columns)
    
    X_area = X_features[["log_area"]]
    X_climate = X_features[climate_predictors]
    
    # Generate interaction terms among climate variables
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_climate_poly = poly.fit_transform(X_climate)
    feature_names = poly.get_feature_names_out(X_climate.columns)
    X_climate_poly = pd.DataFrame(X_climate_poly, columns=feature_names)
    X_log_a_climate_poly = X_climate_poly.multiply(X_area.values)
    X_log_a_climate_poly.columns = [f'{col}_x_log_area' for col in feature_names]
    X = pd.concat([X_area, X_climate_poly, X_log_a_climate_poly], axis=1)
    assert X.shape[1] == 2*(math.comb(len(climate_predictors), 2) + len(climate_predictors)) + 1

    if target_scaler is None:
        target_scaler = StandardScaler()
        target_scaler.fit(gdf[["log_sr"]])

    y = pd.DataFrame(target_scaler.transform(gdf[["log_sr"]]), columns=["log_sr"])

    return X, y, feature_scaler, target_scaler


def evaluate_model_per_hab(dataset, habitats):
    result_all = {}
    climate_predictors = dataset.aggregate_labels
    gdf_full = dataset.gdf.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_cv_partition_idx, test_cv_partition_idx = train_test_split(
        gdf_full.partition.unique(), test_size=0.2, random_state=42
    )
    

    for hab in habitats:
        gdf = gdf_full[gdf_full.habitat_id == hab].reset_index(drop=True)
        gdf_train = gdf[gdf.partition.isin(train_cv_partition_idx)]
        gdf_test = gdf[gdf.partition.isin(test_cv_partition_idx)]

        X_train, y_train, feature_scaler, target_scaler = get_Xy(gdf_train, climate_predictors)
        X_test, y_test, _, _ = get_Xy(gdf_test, climate_predictors, feature_scaler, target_scaler)

        
        reg = Ridge(**CONFIG["ridge"])
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)


        result_all[hab] = {}
        result_all[hab]["reg"] = reg
        result_all[hab]["X_test"] = X_test
        result_all[hab]["y_test"] = y_test
        result_all[hab]["y_pred"] = y_pred
        result_all[hab]["mse"] = mse


    return result_all


def compute_logc_z(res, climate_predictors):
    """
    Computes the estimated intercept and slope for each sample in the dataset.

    Args:
        model (LinearRegression): Trained linear regression model.
        X (pd.DataFrame): Features (could be training or test features).
        data_original (pd.DataFrame): Original data including environmental variables.

    Returns:
        intercepts (np.ndarray): Estimated intercepts for each sample.
        slopes (np.ndarray): Estimated slopes for each sample.
        
    $$
    \log SR = [\beta_0 + \sum_i \gamma_i x_i] +  \log A [\beta_1 + \sum_i \delta_i x_i]
    $$
    """
    for hab in res:
        # Get model coefficients
        model = res[hab]["reg"]
        X = res[hab]["X_test"]
        coef, = model.coef_
        intercept = model.intercept_
        
        # Identify indices of features
        env_var_names = [col for col in X.columns if "area" not in col]
        log_a_x_env_var_names = [col for col in X.columns if "x_log_area" in col]
        
        # assert order
        for ev, inter in zip(env_var_names, log_a_x_env_var_names):
            assert ev in inter
        
        gamma_i_indices = [X.columns.get_loc(col) for col in env_var_names]
        delta_i_indices = [X.columns.get_loc(col) for col in log_a_x_env_var_names]
        
        log_area_idx = X.columns.get_loc("log_area")
        
        # Compute intercepts and slopes
        gamma_i = coef[gamma_i_indices]
        delta_i = coef[delta_i_indices]
        z_i = X[env_var_names].values
        logc = intercept + np.dot(z_i, gamma_i)
        z = coef[log_area_idx] + np.dot(z_i, delta_i)
        res[hab]["logc"] = logc
        res[hab]["z"] = z
        test_slope_intercept_calculation(model, X, logc, z)
    return res

def test_slope_intercept_calculation(model, X_test, logc, z):
    log_area = X_test['log_area'].values
    y_pred_manual = logc + z * log_area
    y_pred_model = model.predict(X_test).flatten()
    assert np.allclose(y_pred_manual, y_pred_model, atol=1e-6)


if __name__ == '__main__':
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]

    if True:
        dataset = figure_2_Ridge.process_results()

    # Fitting models with spatial block cross-validation
    result_modelling = evaluate_model_per_hab(dataset, habitats)
    compute_logc_z(result_modelling, dataset.aggregate_labels)
    
    fig, axes = plt.subplots(5,2,figsize=(8, 10), 
                            #  sharex=True, sharey=True
                             )
    for i, hab in enumerate(habitats):
        ax = axes.flatten()[i]
        logc = result_modelling[hab]["logc"]
        z = result_modelling[hab]["z"]
        mse = result_modelling[hab]["mse"]
        y_pred = result_modelling[hab]["y_pred"]
        y_test = result_modelling[hab]["y_test"]
        X_test = result_modelling[hab]["X_test"]

        ax.scatter(logc, z, alpha=0.5, c=X_test.log_area)
        ax.set_xlabel('logc')
        ax.set_ylabel('z')
        ax.set_title(hab)
    fig.tight_layout()
    fig.savefig(Path(__file__).stem + ".png", dpi=300)



    # fig, ax = plt.subplots(1,figsize=(8, 6))
    # ax.scatter(X_test.log_area, y_test, alpha=0.7)
    
    
