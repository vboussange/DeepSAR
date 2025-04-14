# This file is used generate an anonymised dataset of vegetation plots.
# TODO: you need to discard true species name when publishing the dataset
import hashlib
from collections import defaultdict
from src.data_processing.utils_eva import EVADataset
import base64
from pathlib import Path
import pandas as pd
from eva_species_process import clean_species_name
from tqdm import tqdm

RAW_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/matched/"
RAW_GIFT_DATA = Path(__file__).parent / "../../data/raw/GIFT/"
ANONYMISED_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/anonymised"
ANONYMISED_GIFT_DATA = Path(__file__).parent / "../../data/processed/GIFT/anonymised"
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
eva_plot_df = pd.read_csv(
    "../../data/raw/EVA/172_SpeciesAreaRel20230227_notJUICE_header.csv",
    header=0,
    usecols=[
        "Relevé area (m²)",
        "Expert System",
        "Longitude",
        "Latitude",
        "Location uncertainty (m)",
        "PlotID",
        "Date of recording"
    ],
    sep="\t",
    engine="python",
    index_col="PlotID",
    quoting=3
)
eva_plot_df.columns = ['area_m2',
                    'expert_system',
                    'longitude',
                    'latitude',
                    'location_uncertainty_m',
                    'recording_date']
gift_species_df = pd.read_csv(RAW_GIFT_DATA / "gift_checklists.csv")

# checks
species_eva = set(eva_species_df.gift_matched_species_name.unique())
species_gift = set(clean_species_name(sp) for sp in gift_species_df['work_species'].dropna().unique())
assert species_eva.issubset(species_gift), "Not all EVA species are present in GIFT dataset"

plot_species = set(eva_species_df.plot_id.unique())
assert plot_species.issubset(eva_plot_df.index), "Not all EVA plots are present in EVA plot data"


spid_dict = {}
for species in tqdm(species_gift):
    spid = generate_spid(species)
    if spid in spid_dict.values():
        raise ValueError(f"Duplicate spid '{spid}' generated for species '{species}'")
    spid_dict[species] = spid
print(spid_dict)

# saving the anonymised data as parquet data
eva_species_df['anonymised_species_name'] = eva_species_df['gift_matched_species_name'].map(spid_dict)
gift_species_df['anonymised_species_name'] = gift_species_df['work_species'].map(spid_dict)

# assigning habitats to eva_species_data and gift_species_data
# TODO; we should add as many columns as habitats, and mark them with 1 or 0 should they have appeared once in a plot with corresponding habitat
# it would probably be best if we had were only considering EUNIS habitats at level 1
# maps for these habitats data could be retrieved from https://www.eea.europa.eu/en/datahub/datahubitem-view/220be9b6-bf67-4ea0-b976-65ca57a863b5?activeAccordion=

eva_species_df.to_parquet(ANONYMISED_EVA_DATA / "species_data.parquet")
gift_species_df.to_parquet(ANONYMISED_GIFT_DATA / "species_data.parquet")
# saving the plot data as parquet data
eva_plot_df.to_parquet(ANONYMISED_EVA_DATA / "plot_data.parquet")


# no need to save to parquet gift_plot_data, it is alrady in optimized format (gpkg)