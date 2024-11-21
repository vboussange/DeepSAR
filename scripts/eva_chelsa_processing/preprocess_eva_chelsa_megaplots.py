import pandas as pd
import geopandas as gpd
import numpy as np
from src.plotting import read_result, ResultData
import git 
from pathlib import Path
PATH_AUGMENTED_DATA = Path(__file__).parent / Path("../../data/processed/EVA_CHELSA/")

def process_results(path_results=PATH_AUGMENTED_DATA):
    """
    Reading and processing compiled data.
    """
    results = read_result(path_results)
    gdf_full = results["SAR_data"]
    config = results["config"]
    aggregate_labels = config["env_vars"] + ["std_" + env for env in config["env_vars"]]

    gdf_nonan = gdf_full.dropna(axis=0, how="any")
    print("we have", len(gdf_full), "entries")
    print("we have", len(gdf_nonan), "entries with non nans")
    print("Filtering rows with nans")
    gdf_full = gdf_nonan

    gdf_full["log_area"] = np.log(gdf_full["area"].astype(np.float64))  # area
    gdf_full["log_sr"] = np.log(gdf_full["sr"].astype(np.float64))
    # gdf_full = gdf_full.reset_index(drop=True)

    return ResultData(gdf_full, config, aggregate_labels, None)

def preprocess_gdf_hab(gdf_full, hab, random_state):
    if hab == "all":
        gdf = pd.concat([gdf_full[gdf_full.habitat_id == hab], gdf_full[gdf_full.num_plots == 1]])
    else:
        gdf = gdf_full[gdf_full.habitat_id == hab] 
        
    # taking idx of single plots, filtering out possible doublon
    # plot_idxs = plot_gdf['plot_idxs'].value_counts()[plot_gdf['plot_idxs'].value_counts() == 1].index
    plot_idxs = gdf[gdf.num_plots == 1].index.to_series()
    megaplot_idxs = gdf[gdf.num_plots > 1].index.to_series()
    print(hab)
    print(f"Number of raw plots = {len(plot_idxs)}")
    print(f"Number of mega plots = {len(megaplot_idxs)}")
    megaplot_idxs = megaplot_idxs.sample(min(len(plot_idxs), len(megaplot_idxs)), random_state=random_state, replace=False)
    plot_megaplot_gdf_hab = gdf_full.loc[megaplot_idxs]
    return plot_megaplot_gdf_hab.sample(frac=1, random_state=random_state)

def load_preprocessed_data(hab, git_hash, random_state, path_augmented_data=PATH_AUGMENTED_DATA):
    path_preprocess_data = path_augmented_data / f"{hab}_preprocessed_data_random_state_{random_state}_{git_hash}.pkl"
    return pd.read_pickle(path_preprocess_data)


if __name__ == "__main__":
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    random_state = 1
    dataset = process_results()
    
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)
    for hab in habitats:
        path_preprocess_data = PATH_AUGMENTED_DATA.parent / f"{hab}_preprocessed_data_random_state_{random_state}_{sha}.pkl"
        gdf = preprocess_gdf_hab(dataset.gdf, hab, random_state)
        gdf.to_pickle(path_preprocess_data)