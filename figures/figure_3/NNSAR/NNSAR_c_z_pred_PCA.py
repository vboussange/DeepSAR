"""
Plotting env features contributions to c and z, only using PCA.

"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pickle
from pathlib import Path

from sklearn.linear_model import LinearRegression


result_path = Path(__file__).parent / Path("../../scripts/NNSAR/NNSAR_fit_simple.pkl")
    
with open(result_path, 'rb') as file:
    results_fit_split = pickle.load(file)["result_modelling"]
        
climate_predictors  = results_fit_split["climate_predictors"]

# Assuming df is the DataFrame with the required data
# Replace this with your actual DataFrame
df = pd.read_csv('NNSAR_c_vs_z.csv')

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

# Step 3: Perform PCA
pca_mean = PCA(n_components=1)
pca_std = PCA(n_components=1)

pca1_mean = pca_mean.fit_transform(mean_data)
pca1_std = pca_std.fit_transform(std_data)

# Step 4: Prepare the dataset for regression
df_pca = pd.DataFrame({
    'PCA1_mean': pca1_mean.flatten(),
    'PCA1_std': pca1_std.flatten(),
    'logc': df['logc_std'],
    'z': df['z_std']
})


# Step 5: Add a constant (intercept term) for the regression
X = sm.add_constant(df_pca[['PCA1_mean', 'PCA1_std']])

# Step 6: Fit the OLS model using statsmodels for both logc and z
model_logc = sm.OLS(df_pca['logc'], X).fit()
model_z = sm.OLS(df_pca['z'], X).fit()

# Step 7: Extract coefficients and standard errors
coeff_logc = model_logc.params[1:]
coeff_z = model_z.params[1:]
std_err_logc = model_logc.bse[1:]
std_err_z = model_z.bse[1:]

# Step 8: Create bar plot of standardized coefficients with error bars
coeffs_df = pd.DataFrame({
    'Variable': ['PCA1_mean', 'PCA1_std'],
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
    lambda row: std_err_logc[0] if 'logc' in row['Effect on'] and row['Variable'] == 'PCA1_mean' else
                std_err_logc[1] if 'logc' in row['Effect on'] and row['Variable'] == 'PCA1_std' else
                std_err_z[0] if 'z' in row['Effect on'] and row['Variable'] == 'PCA1_mean' else
                std_err_z[1],
    axis=1
)

# Plotting
plt.figure()
ax = sns.barplot(x='Variable', y='Standardized Coefficient', hue='Effect on', data=coeffs_df_melted, capsize=0.1)

# Align the error bars with the bar plot
for i in range(len(coeffs_df_melted)):
    # Find the x position of the bars
    x_pos = ax.patches[i].get_x() + ax.patches[i].get_width() / 2
    plt.errorbar(x_pos, coeffs_df_melted['Standardized Coefficient'][i], 
                 yerr=coeffs_df_melted['Std Error'][i], fmt='none', c='black', capsize=5)

plt.title('Standardized Effect of PCA1 (Mean and Std) on logc and z with Standard Error')
plt.show()