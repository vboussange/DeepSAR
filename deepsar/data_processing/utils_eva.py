import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
import json
import os

# Default base paths with environment variable support
EVA_DATA_DIR = Path(__file__).parent / "../../data/processed/EVA/"

class EVADataset:
    """
    Loader and preprocessor for EVA (European Vegetation Archive) vegetation plot data.
    
    Handles loading of species occurrence data and plot metadata, with efficient
    preprocessing for large-scale datasets (500k+ plots, 20k+ species).
    """
    
    def __init__(self, data_dir=EVA_DATA_DIR):
        self.data_dir = Path(data_dir)
        
        # Cache path for preprocessed arrays
        self.preprocessed_cache = self.data_dir / "anonymised/preprocessed_cache.npz"

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
            species_gdf = species_df.groupby("plot_id")
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
                - species_dict: dict mapping plot_id to list of species names
        """
        plot_data = self.read_plot_data()
        species_data = self.read_species_data()
        return plot_data, species_data
    
    def load_species_matrix(self, use_cache=True):
        """
        Converts GeoPandas data and species dict into array formats for deep learning.
        Optimized for large datasets (500k+ plots, 20k+ species).
        
        Args:
            plot_gdf: GeoDataFrame with plot data. If None, loads from disk.
            species_dict: Dictionary mapping plot_id to species lists. If None, loads from disk.
            use_centroids: If True, use polygon centroids for coordinates (default: True).
            use_cache: Whether to use cached preprocessed data if available (default: True).
        
        Returns:
            tuple: (coords, obs_areas, species_matrix, species_list) where:
                - coords: np.array (N_plots, 2) [x, y] coordinates in float32
                - obs_areas: np.array (N_plots,) observed areas in float32
                - species_matrix: np.array (N_plots, N_species) bool presence/absence
                - species_list: list of species names corresponding to matrix columns
        """
        # Check cache first
        if use_cache and self.preprocessed_cache.exists():
            print(f"Loading preprocessed data from cache: {self.preprocessed_cache}")
            with np.load(self.preprocessed_cache, allow_pickle=True) as data:
                coords = data['coords']
                obs_areas = data['obs_areas']
                species_matrix = data['species_matrix']
                species_list = data['species_list'].tolist()
                
                return coords, obs_areas, species_matrix, species_list
        
        print("Loading plot and species data...")
        plot_gdf, species_dict = self.load_species_dict()
        
        coords = np.column_stack((
                plot_gdf.geometry.x.values,
                plot_gdf.geometry.y.values
            )).astype(np.float32)
        
        # 2. Extract observed areas
        if 'observed_area' in plot_gdf.columns:
            obs_areas = plot_gdf['observed_area'].values.astype(np.float32)
        else:
            raise KeyError("Column 'observed_area' not found in plot data.")
        
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
        # ~1.25 GB for 500k plots x 20k species as bool
        species_matrix = np.zeros((n_plots, n_species), dtype=bool)
        
        # Fill matrix efficiently
        # Map plot_gdf index to species_dict keys
        if hasattr(plot_gdf, 'index'):
            # Use plot_id column if available, otherwise use index
            if 'plot_id' in plot_gdf.columns:
                plot_ids = plot_gdf['plot_id'].tolist()
            else:
                plot_ids = plot_gdf.index.tolist()
        else:
            plot_ids = list(range(len(plot_gdf)))
        
        # Vectorized filling using list comprehension for speed
        print("Filling species matrix (this may take a moment for large datasets)...")
        for row_idx, plot_id in enumerate(tqdm(plot_ids, desc="Processing plots")):
            if plot_id in species_dict:
                species_list = species_dict[plot_id]
                # Vectorized index lookup and assignment
                sp_indices = [species_to_idx[sp] for sp in species_list if sp in species_to_idx]
                if sp_indices:  # Only assign if we have valid species
                    species_matrix[row_idx, sp_indices] = True
        
        # Cache the preprocessed data
        if use_cache:
            self.preprocessed_cache.parent.mkdir(parents=True, exist_ok=True)
            print(f"Caching preprocessed data to {self.preprocessed_cache}")
            np.savez_compressed(
                self.preprocessed_cache,
                coords=coords,
                obs_areas=obs_areas,
                species_matrix=species_matrix,
                species_list=np.array(all_species, dtype=object)
            )
            print(f"✓ Cache saved ({self.preprocessed_cache.stat().st_size / 1e6:.1f} MB)")
        
        return coords, obs_areas, species_matrix, all_species

if __name__ == "__main__":
    dataset = EVADataset()
    df_sp = dataset.read_species_data()
    plot_data, species_dict = dataset.load_species_dict()
    coords, obs_areas, species_matrix, all_species = dataset.load_species_matrix()
