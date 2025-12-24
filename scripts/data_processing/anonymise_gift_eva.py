"""
Anonymizes species names in EVA and GIFT datasets.

This script performs the following operations:
1. Generates unique, deterministic 6-character IDs (spid) for each species name.
2. Replaces original species names with these IDs in both datasets.
3. Renames columns to standard names (record_id, anonymised_species_name).
4. Saves the anonymized datasets to parquet files.
"""

import hashlib
import base64
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import numpy as np
import shutil

# Constants
BASE_DIR = Path(__file__).parent / "../../data/processed"
RAW_EVA_DATA = BASE_DIR / "EVA/preprocessing/"
RAW_GIFT_DATA = BASE_DIR / "GIFT/preprocessing/"
ANONYMISED_EVA_DATA = BASE_DIR / "EVA/anonymised"
ANONYMISED_GIFT_DATA = BASE_DIR / "GIFT/anonymised"

def setup_directories():
    """Cleans and creates output directories."""
    for directory in [ANONYMISED_EVA_DATA, ANONYMISED_GIFT_DATA]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

def generate_spid(species_name: str) -> str:
    """
    Generate a deterministic unique species ID (spid) from a species name.
    
    Args:
        species_name (str): The original species name.
    
    Returns:
        str: A 7-character anonymized species ID.
    """
    hash_bytes = hashlib.sha256(species_name.encode()).digest()
    # Use base32 to avoid confusing characters, take first 7 chars
    spid = base64.b32encode(hash_bytes).decode('utf-8')[:7]
    return spid

def validate_datasets(eva_species_df, gift_species_df, eva_plot_df):
    """Checks consistency between datasets."""
    species_eva = set(eva_species_df.species_name.unique())
    species_gift = set(gift_species_df['species_name'].unique())
    assert species_eva.issubset(species_gift), "Not all EVA species are present in GIFT dataset"

    eva_record_ids = set(eva_species_df.record_id.unique())
    assert eva_record_ids.issubset(eva_plot_df.record_id), "Not all EVA records are present in EVA plot data"



if __name__ == "__main__":
    setup_directories()

    print("Loading data...")
    eva_species_df = pd.read_parquet(RAW_EVA_DATA / "species_data.parquet")
    gift_species_df = pd.read_parquet(RAW_GIFT_DATA / "species_data.parquet")
    gift_plot_df = gpd.read_parquet(RAW_GIFT_DATA / "plot_data.parquet").to_crs("EPSG:3035")
    eva_plot_df = gpd.read_parquet(RAW_EVA_DATA / "plot_data.parquet").to_crs("EPSG:3035")

    print("Validating data consistency...")
    validate_datasets(eva_species_df, gift_species_df, eva_plot_df)

    print("Generating anonymized IDs...")
    species_gift = set(gift_species_df['species_name'].unique())
    spid_dict = {}
    for species in tqdm(species_gift, desc="Hashing species"):
        spid = generate_spid(species)
        if spid in spid_dict.values():
            raise ValueError(f"Duplicate spid '{spid}' generated for species '{species}'")
        spid_dict[species] = spid
    
    # Anonymizing species names
    print("Anonymizing datasets...")
    eva_species_df['anonymised_species_name'] = eva_species_df['species_name'].map(spid_dict)
    if eva_species_df['anonymised_species_name'].isna().any():
        raise ValueError("Some species in EVA dataset could not be anonymized. Check for missing mappings.")

    gift_species_df['anonymised_species_name'] = gift_species_df['species_name'].map(spid_dict)
    if gift_species_df['anonymised_species_name'].isna().any():
        raise ValueError("Some species in GIFT dataset could not be anonymized. Check for missing mappings.")

    # Saving data
    print("Saving anonymized data...")
    
    # EVA
    eva_species_out = eva_species_df[["record_id", "anonymised_species_name", "cover_percent"]]
    eva_species_out.to_parquet(ANONYMISED_EVA_DATA / "species_data.parquet")
    
    eva_plot_out = eva_plot_df[["record_id", "recording_date", "area_m2", "location_uncertainty_m", "geometry"]]
    eva_plot_out.to_parquet(ANONYMISED_EVA_DATA / "plot_data.parquet")
    
    # GIFT
    gift_species_out = gift_species_df[["record_id", "anonymised_species_name"]]
    gift_species_out.to_parquet(ANONYMISED_GIFT_DATA / "species_data.parquet")
    
    gift_plot_out = gift_plot_df[["record_id", "geometry"]]
    gift_plot_out.to_parquet(ANONYMISED_GIFT_DATA / "plot_data.parquet")
    
    print("Anonymisation completed successfully.")
