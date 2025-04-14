import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from pathlib import Path
from math import radians
import geopandas as gpd
from tqdm import tqdm
import pickle
import xarray as xr

from src.data_processing.utils_landcover import EUNISDataset

GIFT_DATA_DIR = Path(__file__).parent / "../../data/processed/GIFT/anonymised"

class GIFTDataset:
    def __init__(self, data_dir=GIFT_DATA_DIR):
        self.data_dir = data_dir

    def read_species_data(self):
        species_data = self.data_dir / "species_data.parquet"
        if species_data.exists():
            return pd.read_parquet(species_data)
        else:
            raise FileNotFoundError(f"Anoymised species data at {GIFT_DATA_DIR.resolve()} not found, did you download/anonymise the data?")
    
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
