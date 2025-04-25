# TODO: including all steps related in eva_anonymisation.py
import pandas as pd
import re
import numpy as np
from difflib import get_close_matches
from tqdm import tqdm
from pathlib import Path
import geopandas as gpd
from eva_preprocessing import clean_species_name
from src.data_processing.utils_env_pred import EXTENT_DATASET

# ---------------------- CONFIGURATION ---------------------- #
OUTPUT_FOLDER = Path(__file__).parent / "../../data/processed/GIFT/preprocessing"

PROCESSED_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/preprocessing/"
RAW_GIFT_DATA = Path(__file__).parent / "../../data/raw/GIFT"

eva_species_df = pd.read_parquet(PROCESSED_EVA_DATA / "species_data.parquet")
gift_species_df = pd.read_csv(RAW_GIFT_DATA / "species_data.csv")
gift_plot_df = gpd.read_file(RAW_GIFT_DATA / "plot_data.gpkg")
eva_plot_df = gpd.read_file(PROCESSED_EVA_DATA / "plot_data.gpkg")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

gift_species_df['work_species_cleaned'] = gift_species_df['work_species'].apply(clean_species_name)

# Filtering out all non resolved species level entries
gift_species_df = gift_species_df[~gift_species_df["work_species_cleaned"].str.contains("spec.", na=False)]


# # Crop plot_gdf to the extent of climate_raster
print("Cropping plot_gdf to the extent of climate_raster...")
gift_plot_df = gift_plot_df.cx[EXTENT_DATASET[0]:EXTENT_DATASET[2], EXTENT_DATASET[1]:EXTENT_DATASET[3]]

# creating habitat specific plots from GIFT data
plot_id = 0
filtered_gift_species_df = []
filtered_gift_plot_df = []
for hab in ["Q", "R", "S", "T", "all"]:
    # get all species in the habitat
    if hab == "all":
        eva_plot_id_hab = eva_plot_df[eva_plot_df["level_1"].isin(["Q", "R", "S", "T"])].plot_id.unique() #this should be equal to eva_plot_df.plot_id.unique()
    else:
        eva_plot_id_hab = eva_plot_df[eva_plot_df["level_1"] == hab].plot_id.unique()
    hab_species = eva_species_df[eva_species_df["plot_id"].isin(eva_plot_id_hab)].gift_matched_species_name.unique()
    gift_sp_df = gift_species_df[gift_species_df["work_species_cleaned"].isin(hab_species)]
    gift_old_plot_ids = gift_sp_df["entity_ID"].dropna().unique()
    gift_new_plot_ids = {old_id: plot_id+i for (i, old_id) in enumerate(gift_old_plot_ids)}
    gift_sp_df["entity_ID"] = gift_sp_df["entity_ID"].map(gift_new_plot_ids)
    gift_pl_df = gift_plot_df[gift_plot_df["entity_ID"].isin(gift_old_plot_ids)]
    gift_pl_df["entity_ID"] = gift_pl_df["entity_ID"].map(gift_new_plot_ids)
    gift_pl_df["level_1"] = hab
    filtered_gift_plot_df.append(gift_pl_df)
    filtered_gift_species_df.append(gift_sp_df)
    plot_id = plot_id + len(gift_old_plot_ids)
    print("Habitat: ", hab, "has ", len(gift_old_plot_ids), "plots", "and ", len(gift_sp_df.work_species_cleaned.unique()), "species")
gift_species_df = pd.concat(filtered_gift_species_df, ignore_index=True)
gift_plot_df = pd.concat(filtered_gift_plot_df, ignore_index=True)

assert len(gift_plot_df) == len(gift_plot_df["entity_ID"].unique())
assert set(gift_plot_df.entity_ID).issubset(set(gift_species_df.entity_ID)), " plot_gdf.entity_ID is not a subset of gift_species_df.entity_ID"


gift_species_df.to_parquet(OUTPUT_FOLDER / "species_data.parquet")
gift_plot_df.to_file(OUTPUT_FOLDER / "plot_data.gpkg", driver="GPKG")