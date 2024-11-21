"""
Plotting c and z vs het and mean env cond. This script implements a quick and
dirty solution where a linear model is fitted on c and z, instead of calculating
contriutions of features directly.
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


def get_Xy(gdf, climate_predictors):
    gdf.reset_index(drop=True, inplace=True)
    
    X_area = gdf[["log_area"]]
    X_climate = gdf[climate_predictors]
    
    # Generate interaction terms among climate variables
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_climate_poly = poly.fit_transform(X_climate)
    feature_names = poly.get_feature_names_out(X_climate.columns)
    X_climate_poly = pd.DataFrame(X_climate_poly, columns=feature_names)
    X_log_a_climate_poly = X_climate_poly.multiply(X_area.values)
    X_log_a_climate_poly.columns = [f'{col}_x_log_area' for col in feature_names]
    X = pd.concat([X_area, X_climate_poly, X_log_a_climate_poly], axis=1)
    assert X.shape[1] == 2*(math.comb(len(climate_predictors), 2) + len(climate_predictors)) + 1


    y = gdf["log_sr"]

    return X, y


def evaluate_model_per_hab(dataset, habitats):
    result_all = {}
    climate_predictors = dataset.aggregate_labels
    gdf_full = dataset.gdf.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_cv_partition_idx, test_cv_partition_idx = train_test_split(
        gdf_full.partition.unique(), test_size=0.2, random_state=42
    )
    result_all["climate_predictors"] = climate_predictors

    for hab in habitats:
        gdf = gdf_full[gdf_full.habitat_id == hab].reset_index(drop=True)
        train_idx = gdf.partition.isin(train_cv_partition_idx)
        gdf_train = gdf[train_idx]
        gdf_test = gdf[~train_idx]

        X_train, y_train = get_Xy(gdf_train, climate_predictors)
        X_test, y_test = get_Xy(gdf_test, climate_predictors)

        
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)


        result_all[hab] = {}
        result_all[hab]["reg"] = reg
        result_all[hab]["gdf"] = gdf.copy()
        result_all[hab]["mse"] = mse
        result_all[hab]["train_idx"] = train_idx
        result_all[hab]["scalers"] = feature_scaler, target_scaler


    return result_all


def calculate_sr_logc_z(reg, X):
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
    coef = reg.coef_
    intercept = reg.intercept_
    
    # Identify indices of features
    env_var_names = [col for col in X.columns if "area" not in col]
    log_a_x_env_var_names = [col for col in X.columns if "x_log_area" in col]
    
    # assert order
    for ev, inter in zip(env_var_names, log_a_x_env_var_names):
        assert ev in inter
    
    gamma_i_indices = [X.columns.get_loc(col) for col in env_var_names]
    delta_i_indices = [X.columns.get_loc(col) for col in log_a_x_env_var_names]
    
    log_area_idx = X.columns.get_loc("log_area")
    
    sr = reg.predict(X)
    
    # Compute intercepts and slopes
    gamma_i = coef[gamma_i_indices]
    delta_i = coef[delta_i_indices]
    z_i = X[env_var_names].values
    logc = intercept + np.dot(z_i, gamma_i)
    z = coef[log_area_idx] + np.dot(z_i, delta_i)
        
    test_slope_intercept_calculation(reg, X, logc, z)
    return sr, logc, z

def test_slope_intercept_calculation(model, X, logc, z):
    log_area = X['log_area'].values
    y_pred_manual = logc + z * log_area
    y_pred_model = model.predict(X).flatten()
    assert np.allclose(y_pred_manual, y_pred_model, atol=1e-6)


if __name__ == '__main__':
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]

    if True:
        dataset = figure_2_Ridge.process_results()

    # Fitting models with spatial block cross-validation
    results_fit_split = evaluate_model_per_hab(dataset, habitats)
    
    climate_predictors  = results_fit_split["climate_predictors"]
    hab = "all"
    gdf = results_fit_split[hab]["gdf"]
    gdf_test = gdf[results_fit_split[hab]["train_idx"]]
    reg = results_fit_split[hab]["reg"]
    feature_scaler, target_scaler = results_fit_split[hab]["scalers"]

    X, y, = get_Xy(gdf_test, climate_predictors)
    sr, logc, z = calculate_sr_logc_z(reg, X)
    
    gdf_test["sr"] = sr
    gdf_test["logc"] = logc
    gdf_test["z"] = z
    
    df = gdf_test.copy()
    
    # Step 1: Separate mean and std columns

    mean_cols = [col for col in climate_predictors if not col.startswith('std_') and col not in ['logc', 'z']]
    std_cols = [col for col in climate_predictors if col.startswith('std_')]

    # Step 2: Standardize the data
    scaler = StandardScaler()
    mean_data = scaler.fit_transform(df[mean_cols])
    std_data = scaler.fit_transform(df[std_cols])


    # Standardize the target variables (logc and z)
    df['logc_std'] = scaler.fit_transform(df[['logc']])
    df['z_std'] = scaler.fit_transform(df[['z']])

    # Step 3: Prepare the dataset for regression
    df_std = pd.DataFrame(mean_data, columns=mean_cols)
    df_std_std = pd.DataFrame(std_data, columns=std_cols)

    # Step 4: Combine mean and std features
    X = pd.concat([df_std, df_std_std], axis=1)

    # Add a constant (intercept term) for the regression
    X = sm.add_constant(X)

    # Step 5: Fit the OLS model using statsmodels for both logc and z
    model_logc = sm.OLS(df['logc_std'], X).fit()
    model_z = sm.OLS(df['z_std'], X).fit()

    # Step 6: Extract coefficients and standard errors
    coeff_logc = model_logc.params[1:]  # Exclude intercept
    coeff_z = model_z.params[1:]        # Exclude intercept
    std_err_logc = model_logc.bse[1:]   # Exclude intercept
    std_err_z = model_z.bse[1:]         # Exclude intercept

    # Step 7: Prepare data for plotting
    coeffs_df = pd.DataFrame({
        'Variable': X.columns[1:],  # Exclude intercept
        'Standardized Effect on logc': coeff_logc,
        'Standardized Effect on z': coeff_z,
        'Std Error on logc': std_err_logc,
        'Std Error on z': std_err_z
    })

    # Melting dataframe for plotting
    coeffs_df_melted = coeffs_df.melt(id_vars="Variable", value_vars=['Standardized Effect on logc', 'Standardized Effect on z'],
                                    var_name="Effect on", value_name="Standardized Coefficient")

    # Add standard error column for error bars
    coeffs_df_melted['Std Error'] = coeffs_df_melted.apply(
        lambda row: std_err_logc[coeffs_df['Variable'].tolist().index(row['Variable'])] if 'logc' in row['Effect on'] else
                    std_err_z[coeffs_df['Variable'].tolist().index(row['Variable'])],
        axis=1
    )

    # Plotting
    # plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Variable', y='Standardized Coefficient', hue='Effect on', data=coeffs_df_melted, capsize=0.1)

    # Align the error bars with the bar plot
    for i in range(len(coeffs_df_melted)):
        # Find the x position of the bars
        x_pos = ax.patches[i].get_x() + ax.patches[i].get_width() / 2
        plt.errorbar(x_pos, coeffs_df_melted['Standardized Coefficient'][i], 
                    yerr=coeffs_df_melted['Std Error'][i], fmt='none', c='black', capsize=5)

    plt.title('Standardized Effect of Mean and Std Features on logc and z with Standard Error')
    plt.xticks(rotation=90)
    plt.show()

    # fig, ax = plt.subplots(1,figsize=(8, 6))
    # ax.scatter(X_test.log_area, y_test, alpha=0.7)
    
    
