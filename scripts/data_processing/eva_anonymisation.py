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
import geopandas as gpd
import numpy as np
from src.data_processing.utils_eunis import extract_habitat_lev1

RAW_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/matched/"
RAW_GIFT_DATA = Path(__file__).parent / "../../data/raw/GIFT/"
ANONYMISED_EVA_DATA = Path(__file__).parent / "../../data/processed/EVA/anonymised"
ANONYMISED_GIFT_DATA = Path(__file__).parent / "../../data/processed/GIFT/anonymised"
ANONYMISED_EVA_DATA.mkdir(parents=True, exist_ok=True)
ANONYMISED_GIFT_DATA.mkdir(parents=True, exist_ok=True)

COUNTRY_DATA = Path(__file__).parent / "../../data/raw/NaturalEarth/ne_10m_admin_0_countries.shp"
COUNTRY_LIST = [
    "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", 
    "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", 
    "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", 
    "Liechtenstein", "Lithuania", "Luxembourg", "North Macedonia", "Malta", 
    "Moldova", "Monaco", "Montenegro", "Netherlands", "Norway", "Poland", 
    "Portugal", "Romania", "San Marino", "Republic of Serbia", 
    "Slovakia", "Slovenia", "Spain", "Sweden", 
    "Switzerland", "Turkey", "Ukraine", "United Kingdom"
]


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

def clean_eva_plots(plot_gdf):
    # calculate SR per plot
    print("Discarding duplicates")
    # identify unique locations and select latest plots
    plot_idx = []
    for _, _gdf in plot_gdf.groupby("geometry"):
        if _gdf["recording_date"].notna().any():
            plot_idx.append(_gdf["recording_date"].idxmax())
        else:
            plot_idx.append(_gdf.plot_id.iloc[np.random.randint(len(_gdf))])

    # filtering for inconsistent coordinates 
    print("Filtering by landcover and extent")
    countries_gdf = gpd.read_file(COUNTRY_DATA)
    eva_countries_gdf = countries_gdf[countries_gdf["SOVEREIGNT"].isin(COUNTRY_LIST)]
    missing_countries = set(COUNTRY_LIST) - set(eva_countries_gdf["SOVEREIGNT"])
    assert len(missing_countries) == 0
    
    _n = len(plot_gdf)
    if plot_gdf.crs != eva_countries_gdf.crs:
        eva_countries_gdf = eva_countries_gdf.to_crs(plot_gdf.crs)
    plot_gdf = plot_gdf.clip(eva_countries_gdf)
    print(f"Discarded {_n - len(plot_gdf)} plots for inconsistent coordinates")
    
    # filtering for uncertainty in meter
    print("Filtering for coordinate uncertainty")
    plot_gdf = plot_gdf[(plot_gdf.location_uncertainty_m.isna()) | (plot_gdf.location_uncertainty_m < 1000)]

    # sorting habitat level
    print("Sorting habitat level")
    plot_gdf["level_1"] = plot_gdf["EUNIS_level"].apply(lambda x: extract_habitat_lev1(x))
    
    # filtering for plot size
    print("Filtering for plot size")
    plot_gdf = plot_gdf[
        ((plot_gdf.level_1.isin(['Q', 'S', 'R'])) & (plot_gdf.area_m2.between(1, 100))) |
        ((plot_gdf.level_1 == 'T') & (plot_gdf.area_m2.between(100, 1000)))
    ]

    return plot_gdf


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
    quoting=3
)
eva_plot_df.rename(columns={
    "Relevé area (m²)": "area_m2",
    "Expert System": "EUNIS_level",
    "Longitude": "longitude",
    "Latitude": "latitude",
    "Location uncertainty (m)": "location_uncertainty_m",
    "PlotID": "plot_id",
    "Date of recording": "recording_date"
}, inplace=True)
gift_species_df = pd.read_csv(RAW_GIFT_DATA / "gift_checklists.csv")

# checks
species_eva = set(eva_species_df.gift_matched_species_name.unique())
species_gift = set(clean_species_name(sp) for sp in gift_species_df['work_species'].dropna().unique())
assert species_eva.issubset(species_gift), "Not all EVA species are present in GIFT dataset"

plot_species = set(eva_species_df.plot_id.unique())
assert plot_species.issubset(eva_plot_df.plot_id), "Not all EVA plots are present in EVA plot data"


# cleaning the plot data
eva_plot_df["geometry"] = gpd.points_from_xy(
    eva_plot_df.longitude, eva_plot_df.latitude, crs="EPSG:4326"
)
eva_plot_df = gpd.GeoDataFrame(eva_plot_df, geometry="geometry", crs="EPSG:4326")
# Convert date strings to datetime objects
eva_plot_df["recording_date"] = pd.to_datetime(eva_plot_df["recording_date"], format="%d.%m.%Y", errors='coerce')
eva_plot_df = clean_eva_plots(eva_plot_df)

# filtering species against plots selected
eva_species_df = eva_species_df[eva_species_df.plot_id.isin(eva_plot_df.plot_id.unique())]


spid_dict = {}
for species in tqdm(species_gift):
    spid = generate_spid(species)
    if spid in spid_dict.values():
        raise ValueError(f"Duplicate spid '{spid}' generated for species '{species}'")
    spid_dict[species] = spid
print(spid_dict)

# saving the anonymised data as parquet data
eva_species_df['anonymised_species_name'] = eva_species_df['gift_matched_species_name'].map(spid_dict)
if eva_species_df['anonymised_species_name'].isna().any():
    raise ValueError("Some species in EVA dataset could not be anonymized. Check for missing mappings in spid_dict.")

gift_species_df['anonymised_species_name'] = gift_species_df['work_species'].apply(clean_species_name).map(spid_dict)
if gift_species_df['anonymised_species_name'].isna().any():
    raise ValueError("Some species in GIFT dataset could not be anonymized. Check for missing mappings in spid_dict.")

# filtering eva plots against species selected
eva_plot_df = eva_plot_df[eva_plot_df.plot_id.isin(eva_species_df.plot_id.unique())]

# assigning habitats to eva_species_data and gift_species_data
# TODO; we should add as many columns as habitats, and mark them with 1 or 0 should they have appeared once in a plot with corresponding habitat
# it would probably be best if we had were only considering EUNIS habitats at level 1
# maps for these habitats data could be retrieved from https://www.eea.europa.eu/en/datahub/datahubitem-view/220be9b6-bf67-4ea0-b976-65ca57a863b5?activeAccordion=

eva_species_df.to_parquet(ANONYMISED_EVA_DATA / "species_data.parquet")
gift_species_df.to_parquet(ANONYMISED_GIFT_DATA / "species_data.parquet")
eva_plot_df.to_file(ANONYMISED_EVA_DATA / "plot_data.gpkg", driver="GPKG")


# no need to save to parquet gift_plot_data, it is alrady in optimized format (gpkg)