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
"""
import pandas as pd
import re
import numpy as np
from difflib import get_close_matches
from tqdm import tqdm
from pathlib import Path


# ---------------------- CONFIGURATION ---------------------- #
EVA_SPECIES_FILE = Path(__file__).parent / "../../data/EVA/raw/172_SpeciesAreaRel20230227_notJUICE_species.csv"
GIFT_CHECKLIST_FILE = Path(__file__).parent / "../../data/GIFT/gift_checklists.csv"
OUTPUT_FOLDER = Path(__file__).parent / "../../data/EVA/processed/"
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
            return cleaned_name
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
        return max(potential_matches, key=lambda x: x[1])[0]

    return "NA"


# ---------------------- DATA LOADING ---------------------- #
def load_data():
    species_eva_df = pd.read_csv(EVA_SPECIES_FILE, sep="\t", engine="python", on_bad_lines='skip')
    gift_df = pd.read_csv(GIFT_CHECKLIST_FILE)

    return species_eva_df, gift_df


# ---------------------- MAIN PROCESSING FUNCTION ---------------------- #
def process_species_matching():
    species_eva_df, gift_df = load_data()

    # Filtering vascular plants
    eva_vascular_df = species_eva_df[species_eva_df["Taxon group"] == "Vascular plant"].copy()

    print(f"Original dataset species count: {species_eva_df['Matched concept'].nunique()}")
    print(f"Filtered vascular dataset rows: {len(eva_vascular_df)}")
    
    # Create a unique dataset of species entries to process
    unique_species_df = eva_vascular_df[FIELDS_PRIORITY].drop_duplicates()
    print(f"Unique species combinations to process: {len(unique_species_df)}")
    
    # Preparing GIFT species set
    gift_species_set = set(gift_df["work_species"].dropna().apply(clean_species_name).unique())
    
    # Matching with GIFT names on the unique dataset only
    tqdm.pandas(desc="Matching unique species")
    unique_species_df["Matched GIFT name"] = unique_species_df.progress_apply(
        lambda row: find_best_match(row, gift_species_set), axis=1
    )
    
    # Merge the matched names back to the full dataset
    eva_vascular_df = eva_vascular_df.merge(
        unique_species_df[FIELDS_PRIORITY + ["Matched GIFT name"]], 
        on=FIELDS_PRIORITY, 
        how="left"
    )
    
    # Logging unmatched cases
    unmatched_df = unique_species_df[unique_species_df["Matched GIFT name"] == "NA"]
    print(f"\nUnmatched unique species: {len(unmatched_df)} / {len(unique_species_df)}")
    
    # Match summary
    matched_unique_species = unique_species_df[unique_species_df["Matched GIFT name"] != "NA"]["Matched GIFT name"].nunique()
    print(f"Matched unique species: {matched_unique_species} / {unique_species_df['Cleaned name'].nunique()}")
    
    # Output DataFrame
    output_df = eva_vascular_df[["Matched GIFT name", "Matched concept", "PlotObservationID"]].rename(
        columns={"Matched GIFT name": "species", "Matched concept": "original_species_name", "PlotObservationID": "plot_id"}
    )
    
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(OUTPUT_FOLDER / 'eva_gift_matched_species.parquet', index=False)
    print(f"\nSaved {len(output_df)} matched entries to {OUTPUT_FOLDER / 'eva_gift_matched_species.parquet'}")


# ---------------------- ENTRY POINT ---------------------- #
if __name__ == "__main__":
    process_species_matching()
