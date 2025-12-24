import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
import json

# Default base paths with environment variable support
EVA_DATA_DIR = Path(__file__).parent / "../../data/processed/EVA/"

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
    Loader and preprocessor for EVA (European Vegetation Archive) vegetation plot data.
    
    Handles loading of species occurrence data and plot metadata, with efficient
    preprocessing for large-scale datasets (500k+ plots, 20k+ species).
    """
    
    def __init__(self, data_dir=EVA_DATA_DIR):
        self.data_dir = Path(data_dir)
        
        # Cache path for preprocessed arrays
        self.preprocessed_cache = self.data_dir / "anonymised/preprocessed_cache.parquet"

    def read_species_data(self):
        species_dataframe_path = self.data_dir / "anonymised/species_data.parquet"
        species_dict_path = self.data_dir / "anonymised/species_data.json"
        if species_dict_path.exists():
            with open(species_dict_path, 'r') as f:
                species_dict = {int(k): v for k, v in json.load(f).items()}
                return species_dict
        elif species_dataframe_path.exists():
            species_dict = {}
            species_df = pd.read_parquet(species_dataframe_path)
            species_gdf = species_df.groupby("record_id")
            species_dict = {}
            for k, v in tqdm(species_gdf, desc="Processing species data"):
                species_dict[k] = list(v["anonymised_species_name"].unique())
            json.dump(species_dict, open(species_dict_path, "w"))
            return species_dict
        else:
            raise FileNotFoundError(f"Anonymised species data not found in {self.data_dir / 'anonymised'}. Did you download/anonymise the data?")
    
    def read_plot_data(self):
        plot_data_file = self.data_dir / "anonymised/plot_data.parquet"
        if plot_data_file.exists():
            plot_data = gpd.read_parquet(plot_data_file)
            return plot_data
        else:
            raise FileNotFoundError(f"Plot data not found at {plot_data_file}. Did you download/anonymise the data?")

    def load_species_dict(self):
        """Load plot and species data.
        
        Returns:
            tuple: (plot_gdf, species_dict) where:
                - plot_gdf: GeoDataFrame with plot metadata and geometry
                - species_dict: dict mapping record_id to list of species names
        """
        plot_data = self.read_plot_data()
        species_data = self.read_species_data()
        return plot_data, species_data
    
    def load_species_matrix(self, use_cache=True):
        """
        Converts GeoPandas data and species dict into a single DataFrame for deep learning.
        Optimized for large datasets (500k+ plots, 20k+ species).
        
        Args:
            use_cache: Whether to use cached preprocessed data if available (default: True).
        
        Returns:
            pd.DataFrame: DataFrame containing:
                - geometry: Point geometry
                - area_m2: float32
                - species_matrix: np.array (N_species,) bool presence/absence per row
                - species_list: list of species names corresponding to matrix columns (metadata)
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
        plot_gdf, species_dict = self.load_species_dict()
        
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
        
        # Get all unique species across the dataset and sort for reproducibility
        all_species = sorted(set(
            species 
            for species_list in species_dict.values() 
            for species in species_list
        ))
        species_to_idx = {sp: i for i, sp in enumerate(all_species)}
        
        n_plots = len(plot_gdf)
        n_species = len(all_species)
        print(f"Dataset size: {n_plots:,} plots × {n_species:,} species")
        
        # Create binary matrix (N_plots x N_species) using bool for memory efficiency
        species_matrix = np.zeros((n_plots, n_species), dtype=bool)
        
        # Fill matrix efficiently
        if hasattr(plot_gdf, 'index'):
            if 'record_id' in plot_gdf.columns:
                plot_ids = plot_gdf['record_id'].tolist()
            else:
                plot_ids = plot_gdf.index.tolist()
        else:
            plot_ids = list(range(len(plot_gdf)))
        
        print("Filling species matrix (this may take a moment)...")
        for row_idx, record_id in enumerate(tqdm(plot_ids, desc="Processing plots")):
            if record_id in species_dict:
                species_list = species_dict[record_id]
                sp_indices = [species_to_idx[sp] for sp in species_list if sp in species_to_idx]
                if sp_indices:
                    species_matrix[row_idx, sp_indices] = True
        
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
    plot_data, species_dict = dataset.load_species_dict()
    df = dataset.load_species_matrix()
    coords = np.column_stack((df.geometry.x, df.geometry.y))
    obs_areas = df['area_m2'].values
    species_list = df.attrs['species_list']
    species_matrix = df[species_list].values
