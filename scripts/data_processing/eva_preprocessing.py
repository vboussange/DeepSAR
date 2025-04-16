"""
This script preprocesses the EVA data at the species entry level by - filtering
for vascular plants - cleaning the species names - matching the species names
with GIFT species names - saving the matched species names to a CSV file

Further processing steps involve 
- anonymisation of the species names
(`eva_anonymisation.py`) 

- preprocessing at the plot level (filtering out for
duplicates, spatial locations, etc.) (`src/utils_eva.py`, but should be placed
in `scripts`)

TODO: include all steps related in `eva_anonymisation.py` in this script
"""
import pandas as pd
import re
import numpy as np
from difflib import get_close_matches
from tqdm import tqdm
from pathlib import Path
import geopandas as gpd

# ---------------------- CONFIGURATION ---------------------- #
EVA_SPECIES_FILE = Path(__file__).parent / "../../data/raw/EVA/172_SpeciesAreaRel20230227_notJUICE_species.csv"
GIFT_CHECKLIST_FILE = Path(__file__).parent / "../../data/raw/GIFT/gift_checklists.csv"
OUTPUT_FOLDER = Path(__file__).parent / "../../data/processed/EVA/matched"
FIELDS_PRIORITY = ["Turboveg2 concept", "Matched concept", "Original taxon concept"]


# ---------------------- UTILITY FUNCTIONS ---------------------- #
def clean_species_name(name):
    # Ensure the input is a string to avoid type errors
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
    # Exact matching
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
        # calculate SR per plot
        print("Discarding duplicates")
        # identify unique locations and select latest plots
        plot_idx = []
        for _, _gdf in plot_gdf.groupby("geometry"):
            if _gdf["recording_date"].notna().any():
                plot_idx.append(_gdf["recording_date"].idxmax())
            else:
                plot_idx.append(_gdf.index[np.random.randint(len(_gdf))])

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
        plot_gdf = plot_gdf[(plot_gdf.uncertainty_m.isna()) | (plot_gdf.uncertainty_m < 1000)]

        # filtering for plot size
        print("Filtering for plot size")
        plot_gdf = plot_gdf[
            ((plot_gdf.Level_1.isin(['Q', 'S', 'R'])) & (plot_gdf.plot_size.between(1, 100))) |
            ((plot_gdf.Level_1 == 'T') & (plot_gdf.plot_size.between(100, 1000)))
        ]

        return plot_gdf
# ---------------------- DATA LOADING ---------------------- #
def load_data():
    eva_species_df = pd.read_csv(EVA_SPECIES_FILE, sep="\t", engine="python", on_bad_lines='skip')
    gift_species_df = pd.read_csv(GIFT_CHECKLIST_FILE)
    
    return eva_species_df, gift_species_df


# ---------------------- MAIN PROCESSING FUNCTION ---------------------- #
def process_species_matching():
    eva_species_df, gift_species_df = load_data()
    # for testing purposes
    # eva_species_df = eva_species_df.sample(1000, random_state=42)

    # Filtering vascular plants
    eva_vascular_df = eva_species_df[eva_species_df["Taxon group"] == "Vascular plant"].copy()

    print(f"Filtered vascular dataset rows: {len(eva_vascular_df)}")
    print(f"Original dataset species count: {eva_species_df['Matched concept'].nunique()}")
    
    # Create a unique dataset of species entries to process
    unique_species_df = eva_vascular_df[FIELDS_PRIORITY].drop_duplicates()
    unique_species_df["Cleaned name"] = unique_species_df["Matched concept"].apply(clean_species_name)
    print(f"Unique species combinations to process: {len(unique_species_df)}")
    
    # Preparing GIFT species set
    gift_species_set = set(gift_species_df["work_species"].dropna().apply(clean_species_name).unique())
    
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
        how="left"
    )
    
    # Logging unmatched cases
    total_eva_species = unique_species_df['Cleaned name'].nunique()
    print(f"Exact match: {unique_species_df.drop_duplicates(subset=['Cleaned name'])['Exact match'].sum()} / {total_eva_species}")
    
    # Match summary
    # Normally, nunique(Matched GIFT name) should be equal to nunique(Cleaned name)
    matched_unique_species = unique_species_df[unique_species_df["Matched GIFT name"] != "NA"]["Cleaned name"].nunique()
    print(f"Approximate match: {matched_unique_species} / {total_eva_species}")
    
    # Output DataFrame
    output_df = eva_vascular_df[["PlotObservationID", "Matched GIFT name", "Matched concept", "Cleaned name", "Exact match"]].rename(
        columns={"PlotObservationID": "plot_id", 
                 "Matched GIFT name": "gift_matched_species_name", 
                 "Cleaned name": "eva_species_name", 
                 "Matched concept" : "eva_original_species_name",
                 "Exact match": "exact_match"}
    )
    # Convert plot_id column to integer type
    output_df["plot_id"] = output_df["plot_id"].astype(int)
    
    # ensure that the merged dataframe has the same number of species as the backbone
    assert output_df["eva_species_name"].nunique() == unique_species_df["Cleaned name"].nunique()
    
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(OUTPUT_FOLDER / 'species_data.parquet', index=False)
    print(f"\nSaved {len(output_df)} matched entries to {OUTPUT_FOLDER / 'species_data.parquet'}")


# ---------------------- ENTRY POINT ---------------------- #
if __name__ == "__main__":
    process_species_matching()
