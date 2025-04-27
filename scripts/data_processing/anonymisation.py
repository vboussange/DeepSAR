# This file is used generate an anonymised dataset of vegetation plots.
# TODO: you need to discard true species name when publishing the dataset
import hashlib
import base64
from pathlib import Path
import pandas as pd
from scripts.data_processing.eva_preprocessing import clean_species_name
from tqdm import tqdm
import geopandas as gpd
import numpy as np

RAW_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/preprocessing/"
RAW_GIFT_DATA = Path(__file__).parent / "../../data/processed/GIFT/preprocessing"
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
gift_plot_df = gpd.read_file(RAW_GIFT_DATA / "plot_data.gpkg")
eva_plot_df = gpd.read_file(RAW_EVA_DATA / "plot_data.gpkg")

# checks
species_eva = set(eva_species_df.gift_matched_species_name.unique())
species_gift =  gift_species_df['work_species_cleaned'].unique()
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

eva_species_df.to_parquet(ANONYMISED_EVA_DATA / "species_data.parquet")
gift_species_df.to_parquet(ANONYMISED_GIFT_DATA / "species_data.parquet")
eva_plot_df.to_file(ANONYMISED_EVA_DATA / "plot_data.gpkg", driver="GPKG")
gift_plot_df.to_file(ANONYMISED_GIFT_DATA / "plot_data.gpkg", driver="GPKG")
print("Anonymisation completed successfully.")