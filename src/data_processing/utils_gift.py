import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from pathlib import Path
from math import radians
import geopandas as gpd
from tqdm import tqdm
import pickle
import xarray as xr
import json

from src.data_processing.utils_landcover import EUNISDataset

GIFT_DATA_DIR = Path(__file__).parent / "../../data/processed/GIFT/preprocessing/unfiltered"

class GIFTDataset:
    def __init__(self, data_dir=GIFT_DATA_DIR):
        self.data_dir = data_dir

    def read_species_data(self):
        species_dataframe_path = self.data_dir / "species_data.parquet"
        species_dict_path = self.data_dir / "species_data.json"
        if species_dict_path.exists():
            with open(species_dict_path, 'r') as f:
                species_dict = {int(k): v for k, v in json.load(f).items()}
                return species_dict
        elif species_dataframe_path.exists():
            species_dict = {}
            species_df = pd.read_parquet(species_dataframe_path)
            species_gdf = species_df.groupby("entity_ID")
            species_dict = {}
            for k, v in tqdm(species_gdf, desc="Processing species data"):
                species_dict[k] = list(v["gift_matched_species_name"].unique())
            json.dump(species_dict, open(species_dict_path, "w"))
            return species_dict
        else:
            raise FileNotFoundError("Anoymised species data not found, did you download/anonymise the data?")
    
    def read_plot_data(self):
        plot_data_file = self.data_dir / "plot_data.gpkg"
        if plot_data_file.exists():
            return gpd.read_file(plot_data_file)
        else:
            raise FileNotFoundError(f"Plot data at {GIFT_DATA_DIR.resolve()} not found, did you download/anonymise the data?")

    def load(self):
            plot_data = self.read_plot_data()
            species_data = self.read_species_data()
            return plot_data, species_data
        
if __name__ == "__main__":
    dataset = GIFTDataset()
    dataset.load()
