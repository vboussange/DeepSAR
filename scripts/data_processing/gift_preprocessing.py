"""
Preprocesses the GIFT data pipeline.

This script performs the following operations:
1. Loads processed EVA data and raw GIFT data.
2. Cleans GIFT species names and filters for resolved species.
3. Crops GIFT plots to the study area extent.
4. Calculates species richness (SR) and observed area for each plot (habitat-agnostic).
5. Creates a filtered dataset where GIFT plots are restricted to species present in the EVA dataset.
6. Saves both unfiltered and filtered datasets to parquet and GeoPackage files.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import geopandas as gpd
from eva_preprocessing import clean_species_name
from deepsar.data_processing.utils_features import EnvironmentalFeatureDataset

# Constants
OUTPUT_FOLDER = Path(__file__).parent / "../../data/processed/GIFT/preprocessing"
PROCESSED_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/preprocessing/"
RAW_GIFT_DATA = Path(__file__).parent / "../../data/raw/GIFT"

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    eva_species_df = pd.read_parquet(PROCESSED_EVA_DATA / "species_data.parquet")
    gift_species_df = pd.read_csv(RAW_GIFT_DATA / "species_data.csv")
    gift_plot_df = gpd.read_file(RAW_GIFT_DATA / "plot_data.gpkg")
    eva_plot_df = gpd.read_parquet(PROCESSED_EVA_DATA / "plot_data.parquet")
    env_features = EnvironmentalFeatureDataset()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Clean GIFT species names
    print("Cleaning GIFT species names...")
    gift_species_df['species_name'] = gift_species_df['work_species'].apply(clean_species_name)

    # Filter out unresolved species (containing 'spec.')
    original_species_count = len(gift_species_df['species_name'].unique())
    gift_species_df = gift_species_df[~gift_species_df["species_name"].str.contains("spec.", regex=False)]
    filtered_species_count = len(gift_species_df['species_name'].unique())
    print(f"Filtered unresolved species: {original_species_count - filtered_species_count} removed, {filtered_species_count} remaining.")

    # Rename entity_ID to record_id
    gift_species_df.rename(columns={"entity_ID": "record_id"}, inplace=True)
    gift_plot_df.rename(columns={"entity_ID": "record_id"}, inplace=True)

    # Alignment validation
    species_eva = set(eva_species_df.species_name.unique())
    species_gift = set(gift_species_df['species_name'].unique())
    assert species_eva.issubset(species_gift), "Not all EVA species are present in GIFT dataset"
    # Filter GIFT to only EVA species
    gift_species_df = gift_species_df[gift_species_df["species_name"].isin(species_eva)]
    print(f"Filtered GIFT species to only those present in EVA: {len(gift_species_df['species_name'].unique())} species remaining.")

    # Crop plots to extent
    print("Cropping plot_gdf to the extent of climate_raster...")
    lc_raster = env_features.load()[1]["landcover"]
    extent_dataset = lc_raster.rio.bounds()
    gift_plot_df = gift_plot_df.to_crs("EPSG:3035")
    gift_plot_df = gift_plot_df.cx[extent_dataset[0]:extent_dataset[2], extent_dataset[1]:extent_dataset[3]]

    # Validation
    assert len(gift_plot_df) == len(gift_plot_df["record_id"].unique())
    assert set(gift_plot_df.record_id).issubset(set(gift_species_df.record_id)), "Plot IDs mismatch between plots and species"

    # Save habitat agnostic data
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    gift_species_df.to_parquet(OUTPUT_FOLDER / "species_data.parquet")
    gift_plot_df.to_parquet(OUTPUT_FOLDER / "plot_data.parquet")
    print(f"Saved GIFT data to {OUTPUT_FOLDER}")
