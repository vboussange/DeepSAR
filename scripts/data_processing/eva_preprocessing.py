"""
Preprocesses the EVA data by filtering for vascular plants, cleaning the species
names, matching the species names with the GIFT backbone, and saving to parquet
dataframes `species_data` and `plot_data`.
"""
import pandas as pd
import re
import numpy as np
from difflib import get_close_matches
from tqdm import tqdm
from pathlib import Path
import geopandas as gpd
from deepsar.data_processing.utils_eunis import extract_habitat_lev1

EVA_SPECIES_FILE = Path(__file__).parent / "../../data/raw/EVA/172_SpeciesAreaRel20230227_notJUICE_species.csv"
GIFT_CHECKLIST_FILE = Path(__file__).parent / "../../data/raw/GIFT/species_data.csv"
OUTPUT_FOLDER = Path(__file__).parent / "../../data/processed/EVA/preprocessing"
FIELDS_PRIORITY = ["Turboveg2 concept", "Matched concept", "Original taxon concept"]

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

def clean_species_name(name):
    """
    Cleans and standardizes species names by removing subspecies, variety,
    cultivar, hybrid, aggregate, uncertainty, and annotation markers, as well as
    content within brackets and extraneous symbols.
    """

    name = str(name)

    cleaned = re.sub(r'\s+subsp\..*$', '', name)    # Remove any text after 'subsp.' (subspecies) and trailing characters
    cleaned = re.sub(r'\s+cf\..*$', '', cleaned)    # Remove any text after 'cf.' (indicating uncertainty in identification)
    cleaned = re.sub(r'\s+aggr\..*$', '', cleaned)    # Remove any text after 'aggr.' (species aggregates)
    cleaned = re.sub(r'\s+var\..*$', '', cleaned)    # Remove any text after 'var.' (variety)
    cleaned = re.sub(r'\s+cv\..*$', '', cleaned)    # Remove any text after 'cv.' (cultivar)
    cleaned = re.sub(r'\s+cfr\..*$', '', cleaned)    # Remove any text after 'cfr.' (comparable to)
    cleaned = re.sub(r'\s+x\s+.*$', '', cleaned)    # Remove hybrid notation marked by ' x ' and following characters
    cleaned = re.sub(r'\([^)]*\)', '', cleaned)    # Remove any content within square brackets []
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)    # Remove any content within square brackets []
    cleaned = re.sub(r'\s*\+.*$', '', cleaned)    # Remove '+' symbols and all following text
    cleaned = re.sub(r'\s+s\.l\.', '', cleaned)    # Remove 's.l.' (sensu lato) annotations indicating broad sense classification
    cleaned = re.sub(r'\s+s\.s\.', '', cleaned)    # Remove 's.s.' (sensu stricto) annotations indicating strict sense classification
    cleaned = re.sub(r'^x[_-]', '', cleaned)    # Remove hybrid prefix notation (x_ or x- at the start)
    cleaned = re.sub(r'\s+x[_-]', ' ', cleaned)    # Remove hybrid prefix notation (' x_' or ' x-') within names
    cleaned = re.sub(r'species', 'spec.', cleaned)    # Replace full word 'species' with the abbreviation 'spec.'

    return cleaned.strip()     # Remove leading and trailing whitespace and return the cleaned name

assert clean_species_name("x_Abies alba subsp. alba s.l. s.s. aggr. (syn)") == "Abies alba"

def find_best_match(row, reference_set: set) -> str:
    """
    Finds the best matching species name from a given row against a reference
    set (GIFT backbone). The function cleans each species name, and checks for an exact match in
    the `reference_set`. If an exact match is found, it returns the matched
    name. If no exact match is found, it attempts to find the closest match
    using string similarity.
    """

    cleaned_names = set()
    for field in FIELDS_PRIORITY:
        cleaned_name = clean_species_name(row[field])
        if cleaned_name in reference_set:
            return cleaned_name, True
        else:
            if cleaned_name != "nan" and cleaned_name not in cleaned_names:
                # Add cleaned name to the set for further processing
                cleaned_names.add(cleaned_name)

    potential_matches = []
    for name in cleaned_names:
        matches = get_close_matches(name, reference_set, n=1, cutoff=0.2)
        if matches:
            match = matches[0]
            confidence = 1.0 - sum(c1 != c2 for c1, c2 in zip(name, match)) / max(len(name), len(match))
            potential_matches.append((match, confidence))

    if potential_matches:
        return max(potential_matches, key=lambda x: x[1])[0], False

    return "NA", False

def clean_eva_plots(plot_gdf):
    """
    Cleans and filters EVA plot GeoDataFrame based on country extent, coordinate
    uncertainty, habitat level, and plot size.
    """

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
    
def load_data():
    eva_species_df = pd.read_csv(EVA_SPECIES_FILE, sep="\t", engine="python", on_bad_lines='skip')
    gift_species_df = pd.read_csv(GIFT_CHECKLIST_FILE)
    eva_plot_df = pd.read_csv(Path(__file__).parent / "../../data/raw/EVA/172_SpeciesAreaRel20230227_notJUICE_header.csv",
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
    eva_plot_df.rename(columns={"Relevé area (m²)": "area_m2",
                                "Expert System": "EUNIS_level",
                                "Longitude": "longitude",
                                "Latitude": "latitude",
                                "Location uncertainty (m)": "location_uncertainty_m",
                                "PlotID": "plot_id",
                                "Date of recording": "recording_date"
                            }, inplace=True)

    
    return eva_species_df, gift_species_df, eva_plot_df
    
if __name__ == "__main__":
    eva_species_df, gift_species_df, eva_plot_df = load_data()
    # for testing purposes
    # eva_species_df = eva_species_df.sample(1000, random_state=42)

    # EVA SPECIES PROCESSING
    # Filtering vascular plants
    eva_vascular_df = eva_species_df[eva_species_df["Taxon group"] == "Vascular plant"].copy()

    print(f"Filtered vascular dataset rows: {len(eva_vascular_df)}")
    print(f"Original dataset species count: {eva_species_df['Matched concept'].nunique()}")
    
    # Create a unique dataset of species entries to process
    unique_species_df = eva_vascular_df[FIELDS_PRIORITY].drop_duplicates()
    unique_species_df["Cleaned name"] = unique_species_df["Matched concept"].apply(clean_species_name)
    
    # Preparing GIFT species set
    gift_species_set = set(gift_species_df["work_species"].dropna().apply(clean_species_name).unique())
    
    # Filtering out all non resolved species level entries
    gift_species_set = {sp for sp in gift_species_set if "spec." not in sp}
    unique_species_df = unique_species_df[~unique_species_df["Cleaned name"].str.contains("spec.", regex=False)]
    print(f"Unique species combinations to process: {len(unique_species_df)}")
    
    # Matching with GIFT names on the unique dataset only
    tqdm.pandas(desc="Matching unique species")
    # Apply function and unpack tuple result into separate columns
    result_tuples = unique_species_df.progress_apply(
        lambda row: find_best_match(row, gift_species_set), axis=1
    )
    
    # Extract the matched name and exact match flag from the tuples
    unique_species_df["Matched GIFT name"] = result_tuples.apply(lambda x: x[0])
    unique_species_df["Exact match"] = result_tuples.apply(lambda x: x[1])
    
    # Merge the matched names back to the full dataset
    eva_vascular_df = eva_vascular_df.merge(
        unique_species_df[FIELDS_PRIORITY + ["Matched GIFT name", "Cleaned name", "Exact match"]], 
        on=FIELDS_PRIORITY, 
        how="left",
    )
    
    # those entries which did not get a match correspond to e.g. genus resolved entries and must be dropped
    eva_vascular_df = eva_vascular_df[~eva_vascular_df["Matched GIFT name"].isna()]
    
    # Logging unmatched cases
    total_eva_species = unique_species_df['Cleaned name'].nunique()
    print(f"Exact match: {unique_species_df.drop_duplicates(subset=['Cleaned name'])['Exact match'].sum()} / {total_eva_species}")
    
    # Match summary
    # Normally, nunique(Matched GIFT name) should be equal to nunique(Cleaned name)
    matched_unique_species = unique_species_df[unique_species_df["Matched GIFT name"] != "NA"]["Cleaned name"].nunique()
    print(f"Approximate match: {matched_unique_species} / {total_eva_species}")
    
    # Output DataFrame
    eva_species_preprocessed = eva_vascular_df[["PlotObservationID", "Matched GIFT name", "Matched concept", "Cleaned name", "Exact match"]].rename(
        columns={"PlotObservationID": "plot_id", 
                 "Matched GIFT name": "gift_matched_species_name", 
                 "Cleaned name": "eva_species_name", 
                 "Matched concept" : "eva_original_species_name",
                 "Exact match": "exact_match"}
    )
    # Convert plot_id column to integer type
    eva_species_preprocessed["plot_id"] = eva_species_preprocessed["plot_id"].astype(int)
    
    # ensure that the merged dataframe has the same number of species as the backbone
    assert eva_species_preprocessed["eva_species_name"].nunique() == unique_species_df["Cleaned name"].nunique()
    
    # EVA PLOTS PROCESING
    # cleaning the EVA plot data
    eva_plot_df["geometry"] = gpd.points_from_xy(
        eva_plot_df.longitude, eva_plot_df.latitude, crs="EPSG:4326"
    )
    eva_plot_df = gpd.GeoDataFrame(eva_plot_df, geometry="geometry", crs="EPSG:4326")
    # Convert date strings to datetime objects
    eva_plot_df["recording_date"] = pd.to_datetime(eva_plot_df["recording_date"], format="%d.%m.%Y", errors='coerce')
    eva_plot_df = clean_eva_plots(eva_plot_df)

    # filtering species against plots selected
    eva_species_preprocessed = eva_species_preprocessed[eva_species_preprocessed.plot_id.isin(eva_plot_df.plot_id.unique())]
    
    # filtering eva plots against species selected
    eva_plot_df = eva_plot_df[eva_plot_df.plot_id.isin(eva_species_preprocessed.plot_id.unique())]

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    eva_species_preprocessed.to_parquet(OUTPUT_FOLDER / 'species_data.parquet', index=False)
    print(f"\nSaved {len(eva_species_preprocessed)} matched entries to {OUTPUT_FOLDER / 'species_data.parquet'}")
    eva_plot_df.to_file(OUTPUT_FOLDER / "plot_data.gpkg", driver="GPKG")
