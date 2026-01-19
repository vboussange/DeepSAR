"""
Preprocesses the EVA data pipeline.

This script performs the following operations:
1. Loads raw EVA species and plot data.
2. Cleans and filters plot data based on location, uncertainty, habitat, and date.
3. Cleans species names and filters for vascular plants.
4. Matches EVA species names against the GIFT backbone taxonomy.
5. Saves the processed species and plot data to parquet files.
"""
import pandas as pd
import re
from difflib import get_close_matches
from tqdm import tqdm
from pathlib import Path
import geopandas as gpd
from deepsar.data_processing.utils_eva import extract_habitat_lev1

# Constants
EVA_SPECIES_FILE = Path(__file__).parent / "../../data/raw/EVA/172_SpeciesAreaRel20230227_notJUICE_species.csv"
GIFT_CHECKLIST_FILE = Path(__file__).parent / "../../data/raw/GIFT/species_data.csv"
OUTPUT_FOLDER = Path(__file__).parent / "../../data/processed/EVA/preprocessing"
FIELDS_PRIORITY = ["turboveg2_concept", "matched_concept", "original_taxon_concept"]

COUNTRY_DATA = Path(__file__).parent / "../../data/raw/NaturalEarth/ne_10m_admin_0_countries.shp"
COUNTRY_LIST = [
    "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", 
    "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Kosovo", "Latvia", 
    "Liechtenstein", "Lithuania", "Luxembourg", "North Macedonia", "Malta", 
    "Moldova", "Monaco", "Montenegro", "Netherlands", "Norway", "Poland", 
    "Portugal", "Romania", "San Marino", "Serbia", 
    "Slovakia", "Slovenia", "Spain", "Sweden", 
    "Switzerland", "Ukraine", "United Kingdom", "Iceland"
]

def clean_species_name(name):
    """
    Standardizes species names by removing subspecies, varieties, hybrids, and annotations.
    """
    name = str(name)
    # Remove infraspecific ranks and annotations
    cleaned = re.sub(r'\s+subsp\..*$', '', name)
    cleaned = re.sub(r'\s+cf\..*$', '', cleaned)
    cleaned = re.sub(r'\s+aggr\..*$', '', cleaned)
    cleaned = re.sub(r'\s+var\..*$', '', cleaned)
    cleaned = re.sub(r'\s+cv\..*$', '', cleaned)
    cleaned = re.sub(r'\s+cfr\..*$', '', cleaned)
    cleaned = re.sub(r'\s+x\s+.*$', '', cleaned)
    
    # Remove brackets and their content
    cleaned = re.sub(r'\([^)]*\)', '', cleaned)
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
    
    # Remove other markers
    cleaned = re.sub(r'\s*\+.*$', '', cleaned)
    cleaned = re.sub(r'\s+s\.l\.', '', cleaned)
    cleaned = re.sub(r'\s+s\.s\.', '', cleaned)
    
    # Handle hybrid prefixes
    cleaned = re.sub(r'^x[_-]', '', cleaned)
    cleaned = re.sub(r'\s+x[_-]', ' ', cleaned)
    
    # Standardize 'species' abbreviation
    cleaned = re.sub(r'species', 'spec.', cleaned)

    return cleaned.strip()

assert clean_species_name("x_Abies alba subsp. alba s.l. s.s. aggr. (syn)") == "Abies alba"

def find_best_match(row, reference_set: set) -> tuple[str, bool]:
    """
    Matches a species entry against a reference set (GIFT backbone).
    
    Returns:
        tuple: (matched_name, is_exact_match)
    """
    cleaned_names = set()
    
    # Check for exact matches in priority order
    for field in FIELDS_PRIORITY:
        cleaned_name = clean_species_name(row[field])
        if cleaned_name in reference_set:
            return cleaned_name, True
        else:
            if cleaned_name != "nan" and cleaned_name not in cleaned_names:
                cleaned_names.add(cleaned_name)

    # Fuzzy matching if no exact match found
    potential_matches = []
    for name in cleaned_names:
        matches = get_close_matches(name, reference_set, n=1, cutoff=0.2)
        if matches:
            match = matches[0]
            # Calculate similarity score
            confidence = 1.0 - sum(c1 != c2 for c1, c2 in zip(name, match)) / max(len(name), len(match))
            potential_matches.append((match, confidence))

    if potential_matches:
        return max(potential_matches, key=lambda x: x[1])[0], False

    return "NA", False

def clean_eva_plots(plot_gdf):
    """
    Filters EVA plots based on spatial extent, uncertainty, habitat, and metadata.
    """
    print("Filtering by landcover and extent...")
    countries_gdf = gpd.read_file(COUNTRY_DATA)
    eva_countries_gdf = countries_gdf[countries_gdf["SOVEREIGNT"].isin(COUNTRY_LIST)]
    
    # Verify all countries are found
    missing_countries = set(COUNTRY_LIST) - set(eva_countries_gdf["SOVEREIGNT"])
    if missing_countries:
        print(f"Warning: Missing countries in shapefile: {missing_countries}")
    
    initial_count = len(plot_gdf)
    if plot_gdf.crs != eva_countries_gdf.crs:
        eva_countries_gdf = eva_countries_gdf.to_crs(plot_gdf.crs)
        
    # Clip plots to study area
    plot_gdf = plot_gdf.clip(eva_countries_gdf)
    print(f"Discarded {initial_count - len(plot_gdf)} plots outside study area")
    
    # Filter by coordinate uncertainty
    print("Filtering for coordinate uncertainty (< 1000m)...")
    plot_gdf = plot_gdf[plot_gdf.location_uncertainty_m < 1000]

    # Extract habitat level
    print("Extracting habitat levels...")
    plot_gdf["level_1"] = plot_gdf["EUNIS_level"].apply(lambda x: extract_habitat_lev1(x))
    
    # Filter by plot size based on habitat type
    print("Filtering for plot size...")
    # Forest/Scrub/Grassland/Heathland: 1-100 m2
    # Woodland (T): 100-1000 m2
    plot_gdf = plot_gdf[
        ((plot_gdf.level_1.isin(['Q', 'S', 'R'])) & (plot_gdf.area_m2.between(1, 100))) |
        ((plot_gdf.level_1 == 'T') & (plot_gdf.area_m2.between(100, 1000)))
    ]
    
    # Filter by recording date
    print("Filtering for recording date (1972-2025)...")
    plot_gdf = plot_gdf[
        (plot_gdf.recording_date.isna()) |
        (plot_gdf.recording_date.dt.year.between(1972, 2025))
    ]

    return plot_gdf
    
def load_data():
    """Loads raw EVA and GIFT datasets."""
    print("Loading datasets...")
    eva_species_df = pd.read_csv(EVA_SPECIES_FILE, sep="\t", engine="python", on_bad_lines='skip')
    # Rename columns to snake_case convention
    eva_species_df.rename(columns={
        "PlotObservationID": "record_id",
        "Taxon group": "taxon_group",
        "Turboveg2 concept": "turboveg2_concept",
        "Matched concept": "matched_concept",
        "Original taxon concept": "original_taxon_concept",
        "Cover %": "cover_percent"
    }, inplace=True)
    
    gift_species_df = pd.read_csv(GIFT_CHECKLIST_FILE)
    
    eva_plot_df = pd.read_csv(
        Path(__file__).parent / "../../data/raw/EVA/172_SpeciesAreaRel20230227_notJUICE_header.csv",
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
        "PlotID": "record_id",
        "Date of recording": "recording_date"
    }, inplace=True)

    # Create GeoDataFrame
    eva_plot_df["geometry"] = gpd.points_from_xy(
        eva_plot_df.longitude, eva_plot_df.latitude, crs="EPSG:4326"
    )
    eva_plot_df = gpd.GeoDataFrame(eva_plot_df, geometry="geometry", crs="EPSG:4326")
    eva_plot_df["recording_date"] = pd.to_datetime(eva_plot_df["recording_date"], format="%d.%m.%Y", errors='coerce')
    
    return eva_species_df, gift_species_df, eva_plot_df

    
if __name__ == "__main__":
    eva_species_df, gift_species_df, eva_plot_df = load_data()
    
    # For testing purposes, uncomment
    # eva_species_df = eva_species_df.sample(1000, random_state=12)
    
    # --- Process Plots ---
    eva_plot_df = clean_eva_plots(eva_plot_df)
    
    # Filter species to only those in selected plots
    eva_species_df = eva_species_df[eva_species_df.record_id.isin(eva_plot_df.record_id.unique())]
    
    # --- Process Species ---
    print("Processing species data...")
    # Filter for vascular plants
    eva_vascular_df = eva_species_df[eva_species_df["taxon_group"] == "Vascular plant"].copy()

    # Create unique backbone of species to process
    eva_backbone = eva_vascular_df[FIELDS_PRIORITY].drop_duplicates()
    
    # Prepare GIFT reference set
    gift_species_set = set(gift_species_df["work_species"].dropna().apply(clean_species_name).unique())
    
    # Match species against GIFT backbone
    tqdm.pandas(desc="Matching unique species")
    result_tuples = eva_backbone.progress_apply(
        lambda row: find_best_match(row, gift_species_set), axis=1
    )
    
    eva_backbone["cleaned_name"] = result_tuples.apply(lambda x: x[0])
    eva_backbone["exact_match"] = result_tuples.apply(lambda x: x[1])
    
    # Merge matched names back to full dataset
    eva_vascular_df = eva_vascular_df.merge(
        eva_backbone[FIELDS_PRIORITY + ["cleaned_name", "exact_match"]], 
        on=FIELDS_PRIORITY, 
        how="left",
    )
    
    # Drop entries without a match (e.g. genus level only), or only resolved at taxon group level
    eva_vascular_df = eva_vascular_df[eva_vascular_df["cleaned_name"] != "NA"]
    eva_vascular_df = eva_vascular_df[~eva_vascular_df["cleaned_name"].str.contains("spec.", regex=False)]

    # Logging
    total_eva_species = eva_backbone['cleaned_name'].nunique()
    exact_matches = eva_backbone.drop_duplicates(subset=['cleaned_name'])['exact_match'].sum()
    print(f"Exact match: {exact_matches} / {total_eva_species}")
    
    matched_unique_species = eva_backbone[eva_backbone["cleaned_name"] != "NA"]["cleaned_name"].nunique()
    print(f"Approximate match: {matched_unique_species} / {total_eva_species}")
    
    # Prepare final output dataframe with standardized column names
    eva_species_final = eva_vascular_df[[
        "record_id", 
        "cleaned_name", 
        "cover_percent",
    ]].copy()
    
    # Rename columns to final output schema
    eva_species_final.rename(columns={
        "cleaned_name": "species_name",
    }, inplace=True)
    
    # Ensure record_id is int
    eva_species_final["record_id"] = eva_species_final["record_id"].astype(int)
            
    # Final plot filtering: remove plots that lost all species during matching
    eva_plot_df = eva_plot_df[eva_plot_df.record_id.isin(eva_species_final.record_id.unique())]

    # Save results
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    species_output_path = OUTPUT_FOLDER / 'species_data.parquet'
    eva_species_final.to_parquet(species_output_path, index=False)
    print(f"\nSaved {len(eva_species_final)} matched entries to {species_output_path}")
    
    plot_output_path = OUTPUT_FOLDER / "plot_data.parquet"
    eva_plot_df.to_parquet(plot_output_path, index=False)
    print(f"Saved {len(eva_plot_df)} plots to {plot_output_path}")
