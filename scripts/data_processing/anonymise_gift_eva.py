"""Anonymising species names in EVA and GIFT datasets."""

import hashlib
import base64
from pathlib import Path
import pandas as pd
from scripts.data_processing.eva_preprocessing import clean_species_name
from tqdm import tqdm
import geopandas as gpd
import numpy as np

RAW_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/preprocessing/"
RAW_GIFT_DATA = Path(__file__).parent / "../../data/processed/GIFT/preprocessing/unfiltered/"
ANONYMISED_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/anonymised"
ANONYMISED_GIFT_DATA = Path(__file__).parent / "../../data/processed/GIFT/anonymised"
if ANONYMISED_EVA_DATA.exists():
    for file in ANONYMISED_EVA_DATA.iterdir():
        file.unlink()
    ANONYMISED_EVA_DATA.rmdir()

if ANONYMISED_GIFT_DATA.exists():
    for file in ANONYMISED_GIFT_DATA.iterdir():
        file.unlink()
    ANONYMISED_GIFT_DATA.rmdir()

ANONYMISED_EVA_DATA.mkdir(parents=True, exist_ok=True)
ANONYMISED_GIFT_DATA.mkdir(parents=True, exist_ok=True)


def generate_spid(species_name: str) -> str:
    """
    Generate a 6-character unique species ID (spid) from a species name.
    
    Args:
        species_name (str): The original species name.
    
    Returns:
        str: A 5-character anonymized species ID.
    """
    hash_bytes = hashlib.sha256(species_name.encode()).digest()
    spid = base64.b32encode(hash_bytes).decode('utf-8')[:7]
    return spid


# loading data
eva_species_df = pd.read_parquet(RAW_EVA_DATA / "species_data.parquet")
gift_species_df = pd.read_parquet(RAW_GIFT_DATA / "species_data.parquet")
gift_plot_df = gpd.read_file(RAW_GIFT_DATA / "plot_data.gpkg").to_crs("EPSG:3035")
eva_plot_df = gpd.read_file(RAW_EVA_DATA / "plot_data.gpkg").to_crs("EPSG:3035")

# checks
species_eva = set(eva_species_df.gift_matched_species_name.unique())
species_gift =  set(gift_species_df['work_species_cleaned'].unique())
assert species_eva.issubset(species_gift), "Not all EVA species are present in GIFT dataset"

plot_species = set(eva_species_df.plot_id.unique())
assert plot_species.issubset(eva_plot_df.plot_id), "Not all EVA plots are present in EVA plot data"


spid_dict = {}
for species in tqdm(species_gift):
    spid = generate_spid(species)
    if spid in spid_dict.values():
        raise ValueError(f"Duplicate spid '{spid}' generated for species '{species}'")
    spid_dict[species] = spid
print(spid_dict)

# anonymisnig species names
eva_species_df['anonymised_species_name'] = eva_species_df['gift_matched_species_name'].map(spid_dict)
if eva_species_df['anonymised_species_name'].isna().any():
    raise ValueError("Some species in EVA dataset could not be anonymized. Check for missing mappings in spid_dict.")

gift_species_df['anonymised_species_name'] = gift_species_df['work_species_cleaned'].map(spid_dict)
if gift_species_df['anonymised_species_name'].isna().any():
    raise ValueError("Some species in GIFT dataset could not be anonymized. Check for missing mappings in spid_dict.")


#
gift_plot_df["area_m2"] = gift_plot_df.geometry.area
gift_plot_df = gift_plot_df.rename(columns={'entity_ID': 'plot_id'}) # to save: plot_id, area_m2, geometry
gift_species_df = gift_species_df.rename(columns={'entity_ID': 'plot_id'}) # to save: plot_id, anonymised_species_name

eva_plot_df = eva_plot_df.rename(columns={'work_ID': 'plot_id'}) # to save: plot_id, recording_data, area_m2, EUNIS_level, location_uncertainty_m, geometry
eva_species_df = eva_species_df.rename(columns={'work_ID': 'plot_id'}) # to save: plot_id, anonymised_species_name


eva_species_df[["plot_id", "anonymised_species_name"]].to_parquet(ANONYMISED_EVA_DATA / "species_data.parquet")
gift_species_df[["plot_id", "anonymised_species_name"]].to_parquet(ANONYMISED_GIFT_DATA / "species_data.parquet")
eva_plot_df[["plot_id", "recording_date", "area_m2", "EUNIS_level", "location_uncertainty_m", "geometry"]].to_parquet(ANONYMISED_EVA_DATA / "plot_data.parquet")
gift_plot_df[["plot_id", "area_m2", "geometry"]].to_parquet(ANONYMISED_GIFT_DATA / "plot_data.parquet")
print("Anonymisation completed successfully.")


# Check the saved files
print("\n=== Checking saved files ===")

# Check EVA files
eva_species_check = pd.read_parquet(ANONYMISED_EVA_DATA / "species_data.parquet")
eva_plot_check = gpd.read_parquet(ANONYMISED_EVA_DATA / "plot_data.parquet")

print(f"EVA species data shape: {eva_species_check.shape}")
print(f"EVA species data columns: {list(eva_species_check.columns)}")
print(f"EVA species data sample:\n{eva_species_check.head()}")

print(f"\nEVA plot data shape: {eva_plot_check.shape}")
print(f"EVA plot data columns: {list(eva_plot_check.columns)}")
print(f"EVA plot data sample:\n{eva_plot_check.head()}")

# Check GIFT files
gift_species_check = pd.read_parquet(ANONYMISED_GIFT_DATA / "species_data.parquet")
gift_plot_check = gpd.read_parquet(ANONYMISED_GIFT_DATA / "plot_data.parquet")

print(f"\nGIFT species data shape: {gift_species_check.shape}")
print(f"GIFT species data columns: {list(gift_species_check.columns)}")
print(f"GIFT species data sample:\n{gift_species_check.head()}")

print(f"\nGIFT plot data shape: {gift_plot_check.shape}")
print(f"GIFT plot data columns: {list(gift_plot_check.columns)}")
print(f"GIFT plot data sample:\n{gift_plot_check.head()}")