import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
import json

# Default base paths with environment variable support
EVA_DATA_DIR = Path(__file__).parents[2] / "data/processed/EVA/"
COUNTRY_DATA = Path(__file__).parents[2] / "data/raw/NaturalEarth/ne_10m_admin_0_countries.shp"
COUNTRY_LIST = [
    "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", 
    "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Kosovo", "Latvia", 
    "Liechtenstein", "Lithuania", "Luxembourg", "North Macedonia", "Malta", 
    "Moldova", "Monaco", "Montenegro", "Netherlands", "Norway", "Poland", 
    "Portugal", "Romania", "San Marino", "Serbia", 
    "Slovakia", "Slovenia", "Spain", "Sweden", 
    "Switzerland", "Ukraine", "United Kingdom", "Iceland"
]

def extract_habitat_lev1(ESyhab: str):
    def is_valid(s: str) -> bool:
        if len(s) >= 2:
            return s[1].isdigit() and s[0].isupper()
        elif len(s) == 1:
            return s.isupper()
        return False

    if ',' in ESyhab:
        parts = [part.strip() for part in ESyhab.split(',')]
        valid_parts = [p for p in parts if is_valid(p)]
        if valid_parts:
            return valid_parts[0][0]
        else:
            return None
    else:
        if is_valid(ESyhab):
            return ESyhab[0]
        else:
            return None
        
class EVADataset:
    """
    Loader and preprocessor for EVA vegetation plot data.

    Handles loading of species occurrence data and plot metadata, with efficient
    preprocessing for large-scale datasets (500k+ plots, 20k+ species).
    """
    
    def __init__(self, data_dir=EVA_DATA_DIR):
        self.data_dir = Path(data_dir)
        
        # Cache path for preprocessed arrays
        self.preprocessed_cache = self.data_dir / "anonymised/preprocessed_cache.parquet"

    def read_species_data(self):
        """Load anonymised EVA species data from parquet.

        Returns:
            pd.DataFrame: Species-level records with `record_id` and
                `anonymised_species_name` columns.
        """
        species_dataframe_path = self.data_dir / "anonymised/species_data.parquet"
        if species_dataframe_path.exists():
            species_df = pd.read_parquet(species_dataframe_path)
            return species_df
        else:
            raise FileNotFoundError(f"Anonymised species data not found in {self.data_dir / 'anonymised'}. Did you download/anonymise the data?")
    
    def read_plot_data(self):
        """Load anonymised EVA plot data from parquet.

        Returns:
            gpd.GeoDataFrame: Plot-level point geometries and metadata.
        """
        plot_data_file = self.data_dir / "anonymised/plot_data.parquet"
        if plot_data_file.exists():
            plot_data = gpd.read_parquet(plot_data_file)
            return plot_data
        else:
            raise FileNotFoundError(f"Plot data not found at {plot_data_file}. Did you download/anonymise the data?")
    
    def load_species_matrix(self, use_cache=True):
        """
        Converts GeoPandas data and species dict into a single DataFrame holding presence-absence matrix.
        
        Args:
            use_cache: Whether to use cached preprocessed data if available (default: True).
        
        Returns:
            gpd.GeoDataFrame: DataFrame containing:
                - geometry: Point geometry, holding location of corresponding sampling site
                - area_m2: float32
                - species columns: bool presence/absence for each species
                - species_list: list of species names corresponding to columns (metadata)
        """
        # Check cache first
        if use_cache and self.preprocessed_cache.exists():
            print(f"Loading preprocessed data from cache: {self.preprocessed_cache}")
            df = gpd.read_parquet(self.preprocessed_cache)
            
            # Identify species columns (all columns except geometry and area_m2)
            species_list = [col for col in df.columns if col not in ['geometry', 'area_m2']]
            df.attrs['species_list'] = species_list
            
            return df
        
        print("Loading plot and species data...")
        plot_gdf = self.read_plot_data()
        species_df = self.read_species_data()
        
        coords = np.column_stack((
                plot_gdf.geometry.x.values,
                plot_gdf.geometry.y.values
            )).astype(np.float32)
        
        # 2. Extract observed areas
        if 'area_m2' in plot_gdf.columns:
            obs_areas = plot_gdf['area_m2'].values.astype(np.float32)
        else:
            raise KeyError("Column 'area_m2' not found in plot data.")
        
        # 3. Vectorize Species Data (One-Hot / Presence-Absence Matrix)
        print("Building species presence-absence matrix...")
        
        # Get all unique species (sorted for consistent ordering)
        all_species = sorted(species_df['anonymised_species_name'].unique().tolist())
        species_to_idx = {sp: i for i, sp in enumerate(all_species)}
        
        # Get record_id to row index mapping from plot_gdf
        record_ids = plot_gdf['record_id'].values
        record_id_to_row = {rid: i for i, rid in enumerate(record_ids)}
        
        # Initialize sparse-friendly approach: build COO-style indices
        n_plots = len(plot_gdf)
        n_species = len(all_species)
        
        # Filter species_df to only include records in plot_gdf
        species_df_filtered = species_df[species_df['record_id'].isin(record_id_to_row)]
        
        # Map record_ids and species to indices
        row_indices = species_df_filtered['record_id'].map(record_id_to_row).values
        col_indices = species_df_filtered['anonymised_species_name'].map(species_to_idx).values
        
        # Build presence-absence matrix efficiently using numpy advanced indexing
        species_matrix = np.zeros((n_plots, n_species), dtype=np.bool_)
        species_matrix[row_indices, col_indices] = True
        
        print(f"  Matrix shape: {species_matrix.shape} ({n_plots} plots × {n_species} species)")
        print(f"  Sparsity: {100 * (1 - species_matrix.sum() / species_matrix.size):.2f}%")
        
        # Construct DataFrame
        df = pd.DataFrame({
            'area_m2': obs_areas,
        })
        df['geometry'] = gpd.points_from_xy(coords[:, 0], coords[:, 1])
        df = gpd.GeoDataFrame(df, geometry='geometry')
        
        # Add species columns
        species_df = pd.DataFrame(species_matrix, columns=all_species)
        df = pd.concat([df, species_df], axis=1)
        
        df.attrs['species_list'] = all_species
        
        # Cache the preprocessed data
        if use_cache:
            self.preprocessed_cache.parent.mkdir(parents=True, exist_ok=True)
            print(f"Caching preprocessed data to {self.preprocessed_cache}")
            df.to_parquet(self.preprocessed_cache, index=False)
            print(f"✓ Cache saved ({self.preprocessed_cache.stat().st_size / 1e6:.1f} MB)")
        
        return df

if __name__ == "__main__":
    # test `extract_habitat_lev1`
    
    examples = [
        'R5',       # → 'R'
        'Sa',       # → None
        'T',        # → None
        'S21',      # → 'S'
        'S21, R23', # → 'S' (both valid, returns first)
        'Sa, T5',   # → 'T' (only T5 is valid)
        'ab, Cd',   # → 'C' (Cd is valid)
        'a',        # → None
    ]
    result = [extract_habitat_lev1(x) for x in examples]
    assert result == ['R', None, "T", 'S', 'S', 'T', None, None], f"Unexpected result: {result}"
    
    # test EvaDataset
    dataset = EVADataset()
    df_sp = dataset.read_species_data()
    plot_data = dataset.read_plot_data()
    df = dataset.load_species_matrix(use_cache=True)
    coords = np.column_stack((df.geometry.x, df.geometry.y))
    obs_areas = df['area_m2'].values
    species_list = df.attrs['species_list']
    species_matrix = df[species_list].values
    print(f"Loaded {len(df)} plots with {len(species_list)} species")
