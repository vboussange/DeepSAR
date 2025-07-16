"""
Preprocesses the GIFT data by cleaning species names, filtering for resolved species-level entries,
matching species with the EVA dataset, calculating species richness and observed area for each plot,
and saving habitat-agnostic and habitat-specific data to parquet and GeoPackage files.
"""

import pandas as pd
import re
import numpy as np
from difflib import get_close_matches
from tqdm import tqdm
from pathlib import Path
import geopandas as gpd
from eva_preprocessing import clean_species_name
from src.data_processing.utils_env_pred import EXTENT_DATASET
from src.data_processing.utils_eunis import EUNISDataset, get_fraction_habitat_landcover

OUTPUT_FOLDER = Path(__file__).parent / "../../data/processed/GIFT/preprocessing"
PROCESSED_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/preprocessing/"
RAW_GIFT_DATA = Path(__file__).parent / "../../data/raw/GIFT"

def clip_GIFT_SR(plot_gdf, gift_species_df, habitat_map):
    """
    Calculates species richness and observed area for each plot.
    """
    plot_gdf["observed_area"] = 0.
    plot_gdf["megaplot_area"] = plot_gdf.geometry.area
    for i, row in plot_gdf.iterrows():
        plot_id = row["entity_ID"]
        clipped_habitat_map = habitat_map.rio.clip([row.geometry], drop=True, all_touched=True)
        proportion_area = get_fraction_habitat_landcover(clipped_habitat_map)
        species = gift_species_df[gift_species_df["entity_ID"] == plot_id]["work_species_cleaned"].values
        sr = len(np.unique(species))
        plot_gdf.loc[i, "sr"] = sr
        plot_gdf.loc[i, "observed_area"] = row["megaplot_area"] * proportion_area

    return plot_gdf


if __name__ == "__main__":
    eva_species_df = pd.read_parquet(PROCESSED_EVA_DATA / "species_data.parquet")
    gift_species_df = pd.read_csv(RAW_GIFT_DATA / "species_data.csv")
    gift_plot_df = gpd.read_file(RAW_GIFT_DATA / "plot_data.gpkg")
    eva_plot_df = gpd.read_file(PROCESSED_EVA_DATA / "plot_data.gpkg")
    eunis = EUNISDataset()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    gift_species_df = gift_species_df[~gift_species_df.work_species.isna()]
    gift_species_df['work_species_cleaned'] = gift_species_df['work_species'].apply(clean_species_name)

    # Filtering out all non resolved species level entries
    gift_species_df = gift_species_df[~gift_species_df["work_species_cleaned"].str.contains("spec.", regex=False)]

    species_eva = set(eva_species_df.gift_matched_species_name.unique())
    species_gift =  set(gift_species_df['work_species_cleaned'].unique())
    assert species_eva.issubset(species_gift), "Not all EVA species are present in GIFT dataset"

    # # Crop plot_gdf to the extent of climate_raster, and reproject to EPSG:3035
    print("Cropping plot_gdf to the extent of climate_raster...")
    gift_plot_df = gift_plot_df.cx[EXTENT_DATASET[0]:EXTENT_DATASET[2], EXTENT_DATASET[1]:EXTENT_DATASET[3]]
    gift_plot_df = gift_plot_df.to_crs("EPSG:3035")

    # compiling habitat agnostic data
    habitat_map = eunis.get_habitat_map("all").where(eunis.raster > -1, np.nan).rio.reproject("EPSG:3035")
    gift_plot_df = clip_GIFT_SR(gift_plot_df, gift_species_df, habitat_map)

    # saving habitat agnostic data
    output_path = OUTPUT_FOLDER / "unfiltered"
    output_path.mkdir(parents=True, exist_ok=True)
    gift_species_df.to_parquet(output_path / "species_data.parquet")
    gift_plot_df.to_file(output_path / "plot_data.gpkg", driver="GPKG")
    print(f"\nSaved GIFT habitat-agnostic data to {output_path / 'species_data.parquet'} and {output_path / 'plot_data.gpkg'}")

    # creating habitat specific plots from GIFT data
    plot_id = 0
    filtered_gift_species_df = []
    filtered_gift_plot_df = []
    for hab in ["all"]:
        # get all species in the habitat
        if hab == "all":
            eva_plot_id_hab = eva_plot_df[eva_plot_df["level_1"].isin(["Q", "R", "S", "T"])].plot_id.unique() #this should be equal to eva_plot_df.plot_id.unique()
        else:
            eva_plot_id_hab = eva_plot_df[eva_plot_df["level_1"] == hab].plot_id.unique()
        # Filter out from the GIFT data all species that are not in the EVA data
        hab_species = eva_species_df[eva_species_df["plot_id"].isin(eva_plot_id_hab)].gift_matched_species_name.unique()
        gift_sp_df = gift_species_df[gift_species_df["work_species_cleaned"].isin(hab_species)]
        # Creating new plot IDs
        gift_old_plot_ids = gift_sp_df["entity_ID"].dropna().unique()
        gift_new_plot_ids = {old_id: plot_id+i for (i, old_id) in enumerate(gift_old_plot_ids)}
        # assigning new plot IDs to the species data
        gift_sp_df["entity_ID"] = gift_sp_df["entity_ID"].map(gift_new_plot_ids)
        # Removing plots which do not have any species from the EVA data, and assigning new plot IDs
        gift_pl_df = gift_plot_df[gift_plot_df["entity_ID"].isin(gift_old_plot_ids)]
        gift_pl_df["entity_ID"] = gift_pl_df["entity_ID"].map(gift_new_plot_ids)
        gift_pl_df["level_1"] = hab
        habitat_map = eunis.get_habitat_map(hab).where(eunis.raster > -1, np.nan).rio.reproject("EPSG:3035")
        gift_pl_df = clip_GIFT_SR(gift_pl_df, gift_sp_df, habitat_map)
        filtered_gift_plot_df.append(gift_pl_df)
        filtered_gift_species_df.append(gift_sp_df)
        plot_id = plot_id + len(gift_old_plot_ids)
        print("Habitat: ", hab, "has ", len(gift_old_plot_ids), "plots", "and ", len(gift_sp_df.work_species_cleaned.unique()), "species")
    gift_species_df = pd.concat(filtered_gift_species_df, ignore_index=True)
    gift_plot_df = pd.concat(filtered_gift_plot_df, ignore_index=True)

    assert len(gift_plot_df) == len(gift_plot_df["entity_ID"].unique())
    assert set(gift_plot_df.entity_ID).issubset(set(gift_species_df.entity_ID)), " plot_gdf.entity_ID is not a subset of gift_species_df.entity_ID"

    output_path = OUTPUT_FOLDER / "filtered"
    output_path.mkdir(parents=True, exist_ok=True)
    gift_species_df.to_parquet(output_path / "species_data.parquet")
    gift_plot_df.to_file(output_path / "plot_data.gpkg", driver="GPKG")
    print(f"\nSaved GIFT habitat-specific data to {output_path / 'species_data.parquet'} and {output_path / 'plot_data.gpkg'}")
