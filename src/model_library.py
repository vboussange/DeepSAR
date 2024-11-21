from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
from src.generate_SAR_data_EVA import get_splot_bio_dfs, kfold_sar_data
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge, RidgeCV
from pathlib import Path
import time
import pandas as pd
import numpy as np

def get_CV_raw_SAR_dataset(habitat, kf, params):
    data_dir = Path(f"../../../data/data_31_03_2023/{habitat}")

    # Load the dataset
    clm5_df, bio_df = get_splot_bio_dfs(data_dir)
    
    print(f"Generating cross validation data for {habitat}")
    start_time = time.time()
    data, cv_idxs = kfold_sar_data(clm5_df, 
                                bio_df, 
                                kf,
                                # npoints=1, 
                                max_aggregate=params["max_aggregate"], 
                                replace=params["replace"], 
                                stats_aggregate=params["stats_aggregate"])
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    # Preprocess and prepare the data
    # Separating features (X) and target variable (y)
    X_train = data.drop(columns=['sr'])  # Features
    y_train = np.log(data['sr'])  # Target variable, log transformed
    
    # log transformation of a
    X_train.a = np.log(X_train.a)
    
    assert not X_train.isna().any().any()
    return X_train, y_train, cv_idxs

def build_models(rfecv=None):
    filter_a = ColumnTransformer([("filter_a", 
                                "passthrough", 
                                ["a"])],
                                remainder="drop")

    poly = PolynomialFeatures(degree=2, interaction_only=True)
        
    MODELS = {"NoPredRidge": make_pipeline(filter_a, StandardScaler(), RidgeCV()),
              "Ridge" : make_pipeline(StandardScaler(), RidgeCV()),
              "RidgeWithInteractions" : make_pipeline(StandardScaler(), poly, RidgeCV()),
              "HGBRPoisson" : HistGradientBoostingRegressor(
                                            loss = "squared_error",
                                            l2_regularization = 1000.,
                                            interaction_cst = None,
                                            # early_stopping=True,
                                            max_depth=5, 
                                            learning_rate=0.1,
                                            # max_iter = 100, 
                                            ),
              "NoPredXGBRegressor" : make_pipeline(filter_a, XGBRegressor(booster="gbtree",
                                        learning_rate=0.2, 
                                        max_depth=6, 
                                        reg_lambda=10.,
                                        objective = "reg:squarederror", # can be reg:squarederror, reg:squaredlogerror
                                        min_child_weight = 1.0,
                                        )),
              "XGBRegressor" : XGBRegressor(booster="gbtree",
                                            learning_rate=0.2, 
                                            max_depth=6, 
                                            reg_lambda=10.,
                                            objective = "reg:squarederror", # can be reg:squarederror, reg:squaredlogerror
                                            min_child_weight = 1.0,
                                            )
                                            }
    if rfecv:
        # /!\ TODO: get inspiration from this post https://stackoverflow.com/questions/31059123/scikit-learn-feature-reduction-using-rfecv-and-gridsearch-where-are-the-coeff
        # Filtering features based on feature selec
        # linear regression with filtered interactions
        name_features = rfecv.get_feature_names_out()
        filter = ColumnTransformer([("filter_rfecv", 
                                    "passthrough", 
                                    name_features)],
                                    remainder="drop")
        MODELS["RidgeWithFilteredInteractions"] = make_pipeline(filter, StandardScaler(), poly, RidgeCV()),
    return MODELS