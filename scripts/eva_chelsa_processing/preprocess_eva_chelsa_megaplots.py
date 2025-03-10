# TODO: normally, this script should not be required anymore, 
# due to the new data augmentation scheme
# TODO: it could be useful to have a function "load_megaplot_data"
# which calculates log transforms, drops na values and returns the data

import pandas as pd
import geopandas as gpd
import numpy as np
from src.plotting import read_result, ResultData
import git 
from pathlib import Path
PATH_FILTERED_AUGMENTED_DATA = Path(__file__).parent / Path("../../data/processed/EVA_CHELSA_filtered/")

# def process_results(random_state, sha):
#     """
#     Reading and preparing raw processed data.
#     """
#     path_results = Path(__file__).parent / Path(f"../../data/processed/EVA_CHELSA/EVA_CHELSA_raw/EVA_CHELSA_raw_random_state_{random_state}_{sha}.pkl")

#     result_path = path_results/f"EVA_CHELSA_raw_random_state_{random_state}_{sha}.pkl"
#     results = read_result(result_path)
#     gdf_full = results["SAR_data"]
#     config = results["config"]
#     aggregate_labels = config["env_vars"] + ["std_" + env for env in config["env_vars"]]

#     gdf_nonan = gdf_full.dropna(axis=0, how="any")
#     print("we have", len(gdf_full), "entries")
#     print("we have", len(gdf_nonan), "entries with non nans")
#     print("Filtering rows with nans")
#     gdf_full = gdf_nonan

#     gdf_full["log_area"] = np.log(gdf_full["area"].astype(np.float64))  # area
#     gdf_full["log_sr"] = np.log(gdf_full["sr"].astype(np.float64))
#     # gdf_full = gdf_full.reset_index(drop=True)

#     return ResultData(gdf_full, config, aggregate_labels, None)

# def preprocess_gdf_hab(gdf_full, hab, random_state):
#     """
#     Preprocessing the data for a specific habitat.
#     """
#     if hab == "all":
#         gdf = pd.concat([gdf_full[gdf_full.habitat_id == hab], gdf_full[gdf_full.num_plots == 1]])
#     else:
#         gdf = gdf_full[gdf_full.habitat_id == hab] 
        
#     # taking idx of single plots, filtering out possible doublon
#     # plot_idxs = plot_gdf['plot_idxs'].value_counts()[plot_gdf['plot_idxs'].value_counts() == 1].index
#     plot_idxs = gdf[gdf.num_plots == 1].index.to_series()
#     megaplot_idxs = gdf[gdf.num_plots > 1].index.to_series()
#     print(hab)
#     print(f"Number of raw plots = {len(plot_idxs)}")
#     print(f"Number of mega plots = {len(megaplot_idxs)}")
#     megaplot_idxs = megaplot_idxs.sample(min(len(plot_idxs), len(megaplot_idxs)), random_state=random_state, replace=False)
#     plot_megaplot_gdf_hab = gdf_full.loc[megaplot_idxs]
#     return plot_megaplot_gdf_hab.sample(frac=1, random_state=random_state)

# def load_preprocessed_data(hab, sha, random_state):
#     path_filtered_augmented_data = PATH_FILTERED_AUGMENTED_DATA / f"{hab}_preprocessed_data_random_state_{random_state}_{sha}.pkl"
#     return pd.read_pickle(path_filtered_augmented_data)

if __name__ == "__main__":
    path_result = Path("/home/boussang/DeepSAR/data/processed/EVA_CHELSA_raw_compilation/EVA_CHELSA_raw_random_state_2_702c19b_checkpoint_all.pkl")
    res = read_result(path_result)
    augmented_data = res["megaplot_data"]
    
    augmented_data["log_area"] = np.log(augmented_data["area"].astype(np.float64))  # area
    
    import matplotlib.pyplot as plt

    # Plot distribution of area with log-scaled bins
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Plot distribution of megaplot area with log-scaled bins
    bins = np.logspace(np.log10(augmented_data["megaplot_area"].min()), np.log10(augmented_data["megaplot_area"].max()), 50)
    ax1.hist(augmented_data["megaplot_area"], edgecolor='k', alpha=0.7, log=True, bins=bins)
    ax1.set_xscale('log')
    ax1.set_title("Distribution of megaplot area")
    ax1.set_xlabel("Megaplot Area")
    ax1.set_ylabel("Frequency")
    ax1.grid(True)

    # Plot distribution of area with log-scaled bins
    bins = np.logspace(np.log10(augmented_data["area"].min()), np.log10(augmented_data["area"].max()), 50)
    ax2.hist(augmented_data["area"], edgecolor='k', alpha=0.7, log=True, bins=bins)
    ax2.set_xscale('log')
    ax2.set_title("Distribution of area")
    ax2.set_xlabel("Area")
    ax2.set_ylabel("Frequency")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs.flatten()[0].scatter(augmented_data["area"], augmented_data["sr"], alpha=0.7, label="Area")
    axs.flatten()[1].scatter(augmented_data["megaplot_area"], augmented_data["sr"], alpha=0.7, label="Megaplot Area", marker='x')
    for ax in axs.flatten():
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("Species Richness vs Area and Megaplot Area")
        ax.set_xlabel("Area / Megaplot Area")
        ax.set_ylabel("Species Richness")
        ax.grid(True)
    plt.show()
    
    
    
    # habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    # random_state = 2
    # dataset = process_results()
    
    # repo = git.Repo(search_parent_directories=True)
    # sha = repo.git.rev_parse(repo.head, short=True)

    # for hab in habitats:
    #     path_filtered_augmented_data = PATH_FILTERED_AUGMENTED_DATA / f"{hab}_preprocessed_data_random_state_{random_state}_{sha}.pkl"
    #     gdf = preprocess_gdf_hab(dataset.gdf, hab, random_state)
    #     gdf.to_pickle(path_filtered_augmented_data)