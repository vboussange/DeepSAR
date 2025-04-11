# This file is used generate an anonymised dataset of vegetation plots.
import hashlib
from collections import defaultdict
from src.data_processing.utils_eva import EVADataset
import base64
from pathlib import Path
import pandas as pd

RAW_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/matched/"
RAW_GIFT_DATA = Path(__file__).parent / "../../data/raw/GIFT/"
ANONYMISED_EVA_DATA = Path(__file__).parent / "../../data/EVA/anonymised"
ANONYMISED_GIFT_DATA = Path(__file__).parent / "../../data/GIFT/anonymised"
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

eva_species_df = pd.read_parquet(RAW_EVA_DATA / "species_data.csv")
eva_plot_df = pd.read_csv(
                "../../data/raw/EVA/172_SpeciesAreaRel20230227_notJUICE_header.csv",
                header=0,
                usecols=[
                    "plot_id",
                    "Level_1",
                    "Level_2",
                    "Level_2_name",
                    "Longitude",
                    "Latitude",
                    "plot_size",
                    "uncertainty_m",
                ],
                sep=",",
                engine="python",
                index_col="plot_id",
            )
gift_species_df = pd.read_csv(RAW_GIFT_DATA / "gift_checklists.csv")

# assigning habitats to EVA species

combined_species = set(eva_species_df.gift_matched_species_name.unique())
combined_species.update(gift_species_df['species'].dropna().unique())
species_list = list(combined_species)

spid_dict = {}
for species in species_list:
    spid = generate_spid(species)
    if spid in spid_dict.values():
        raise ValueError(f"Duplicate spid '{spid}' generated for species '{species}'")
    spid_dict[species] = spid
print(spid_dict)

# saving the anonymised data as parquet data
eva_species_df['anonymised_species_name'] = eva_species_df['gift_matched_species_name'].map(spid_dict)
eva_species_df.to_parquet(ANONYMISED_EVA_DATA / "species_data.parquet")

gift_species_df['anonymised_species_name'] = gift_species_df['work_species'].map(spid_dict)
eva_species_df.to_parquet(ANONYMISED_GIFT_DATA / "species_data.parquet")

# saving the plot data as parquet data
eva_plot_df.to_parquet(ANONYMISED_EVA_DATA / "plot_data.parquet")

# no need to save to parquet gift_plot_data, it is alrady in optimized format (gpkg)