import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import PoissonRegressor, Ridge, RidgeCV
from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from src.utils import HABITATS_ALL, COLORS
import pickle
import os
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures
from src.generate_SAR_data_EVA import generate_SAR_data, get_splot_bio_dfs, format_clm5_for_training
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
from pathlib import Path

def find_extremes(df):
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_df = scaler.transform(df)
    idx1 = 0
    idx2 = 0
    largest_dist = 0
    for i in range(0, len(scaled_df)):
        for j in range(i + 1, len(scaled_df)):
            _dist = euclidean(scaled_df[i, :], scaled_df[j, :])
            if _dist > largest_dist:
                idx1 = i
                idx2 = j
                largest_dist = _dist
    return idx1, idx2, largest_dist    


def fit_model_no_CV(traindata, reg):
    
    # Preprocess and prepare the data
    # Separating features (X) and target variable (y)
    X_train = traindata.drop(columns=['sr'])  # Features
    X_train.a = np.log(X_train.a)

    y_train = np.log(traindata['sr'])  # Target variable

    X_train_model = X_train#[cols_to_keep]

    reg.fit(X_train_model, y_train)
    return reg, X_train_model, y_train, raw_data

def draw_map(ax_map):
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=":")
    ax_map.add_feature(cfeature.LAND, edgecolor="black")
    ax_map.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Overlay the plot locations on the world map
    ax_map.scatter(clm5_df.Longitude, clm5_df.Latitude, c="red", s=1, label="Plot Locations", transform=ccrs.PlateCarree())

    # Add labels and a legend
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    # ax_map.set_title("Plot Locations")
    ax_map.legend(loc="best")
    
def generate_and_plot_predictions(ax_scatter, ax_map, X_train_model, reg, raw_data):
    npoints = 100
    for i, idx in enumerate([10, 2000, 3000]):
        pred1 = pd.DataFrame(np.repeat(raw_data.iloc[idx:idx + 1, :].values, npoints, axis=0),
                            columns=raw_data.columns)
        pred1.a = np.linspace(np.min(X_train_model.a), np.max(X_train_model.a), npoints)
        yhat1 = reg.predict(pred1.drop(columns=["sr"]))
        label = "SAR given \nenv. cond. at site" if i == 0 else None
        ax_scatter.plot(pred1.a, yhat1, label=label, c="tab:purple")
        
        # drawing arrows
        lon, lat = clm5_df.iloc[idx].Longitude, clm5_df.iloc[idx].Latitude
        a, sr = pred1.a[70 + i * 10], yhat1[70+ i * 10] # shifting arrow head for each curve  
        
        # Create a connection patch (arrow) between the two axes
        arrow = patches.ConnectionPatch(
            xyA=(lon, lat), coordsA=ax_map.transData,
            xyB=(a, sr), coordsB=ax_scatter.transData,
            arrowstyle='-|>', shrinkA=5, shrinkB=5,
            mutation_scale=20, color='black',
            transform=ax_map.transData
        )

        fig.patches.append(arrow)
        # Circle the point in the scatter plot
        ax_map.scatter(lon, lat, s=100, facecolors='none', edgecolors='black', marker="s")
        
    # ax_scatter.scatter(raw_data.a, raw_data.sr, label = "Raw data", s = 1.)
    # ax_scatter.scatter(SAR_data.a, SAR_data.sr, label = "Augmented data", s = 1.)
     # idx1, idx2, largest_dist = find_extremes(X_train_model)



params = {"model" : "XGBRegressor", # Model with which are performed 
        "vars" : "avgstd",
        "max_features_to_display": 10,
        "stats_aggregate" : ["mean", "std"]}

models = {"XGBRegressor" : XGBRegressor(booster="gbtree",
                            learning_rate=0.2, 
                            max_depth=6, 
                            reg_lambda=10.,
                            objective = "reg:squarederror", # can be reg:squarederror, reg:squaredlogerror
                            min_child_weight = 1.0,
                            # eta = 0.1,
                            # tree_method = "auto",
                            # verbose=0
                            ),
        # "Ridge" : make_pipeline(RidgeCV())
        }

# plotting feature importance only for single habitat
habitat = "forest_t1"
data_dir = Path(f"../../../data/data_31_03_2023/{habitat}/")
clm5_df, bio_df = get_splot_bio_dfs(data_dir, 
                                        clm5_stats=["avg", "std"])

# SAR generated data
# this can be generated at each epoch
SAR_data = generate_SAR_data(clm5_df, 
                        bio_df, 
                        npoints=len(clm5_df), 
                        max_aggregate=200, 
                        replace=False, 
                        stats_aggregate=params["stats_aggregate"])
# raw data
raw_data = format_clm5_for_training(clm5_df, stats_aggregate=params["stats_aggregate"])
traindata = pd.concat([raw_data,SAR_data])
traindata = pd.concat([raw_data,SAR_data])


reg = models[params["model"]]
reg, X_train_model, y_train, raw_data = fit_model_no_CV(traindata, reg)



# Create a figure and axis with a PlateCarree projection
fig = plt.figure(figsize = (5, 10))
ax_map = fig.add_subplot(212, projection=ccrs.PlateCarree())
draw_map(ax_map)
ax_scatter = fig.add_subplot(211)
ax_scatter.scatter(X_train_model.a, y_train, c="tab:orange", s=2, label = "Training data")

yhat = reg.predict(X_train_model)
ax_scatter.scatter(X_train_model.a, yhat, c="tab:green", s=2, label="Predictions")
ax_scatter.set_yscale("log")
ax_scatter.set_xscale("log")
ax_scatter.set_ylabel("log(Species richnes)")
ax_scatter.set_xlabel("log(Area)")
ax_scatter.set_xlim(7, 12)
ax_scatter.set_ylim(3, 8)
ax_scatter.legend(loc="upper left")
fig.savefig("predicted.pdf", dpi = 300, transparent=True)

generate_and_plot_predictions(ax_scatter, ax_map, X_train_model, reg, raw_data)
ax_scatter.legend(loc="upper left")

fig.savefig("predicted_lowhetero.pdf", dpi = 300, transparent=True)
