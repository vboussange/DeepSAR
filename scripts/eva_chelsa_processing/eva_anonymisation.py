# This file is used generate an anonymised dataset of vegetation plots.
import hashlib
from collections import defaultdict
from src.data_processing.utils_eva import EVADataset
import base64
from pathlib import Path
import pandas as pd

RAW_EVA_DATA = Path(__file__).parent / "../../data/EVA/raw"
ANONYMISED_EVA_DATA = Path(__file__).parent / "../../data/EVA/anonymised"
ANONYMISED_EVA_DATA.mkdir(parents=True, exist_ok=True)

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

species_df = pd.read_csv(RAW_EVA_DATA / "vpl_all.csv",
                header=0,
                usecols=["plot_id", "species"],
                sep=",",
                engine="python")

species_list = species_df.species.unique()
spid_dict = {}
for species in species_list:
    spid = generate_spid(species)
    if spid in spid_dict.values():
        raise ValueError(f"Duplicate spid '{spid}' generated for species '{species}'")
    spid_dict[species] = spid
print(spid_dict)

# saving the anonymised data as parquet data
species_df['species'] = species_df['species'].map(spid_dict)
species_df.to_parquet(ANONYMISED_EVA_DATA / "species_data.parquet")

# saving the plot data as parquet data
plot_df = pd.read_csv(
                RAW_EVA_DATA / "hea_all.csv",
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
plot_df.to_parquet(ANONYMISED_EVA_DATA / "plot_data.parquet")