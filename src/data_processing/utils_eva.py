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

EVA_DATA_DIR = Path(__file__).parent / "../../data/processed/EVA/"

class EVADataset:
    def __init__(self, data_dir=EVA_DATA_DIR):
        self.data_dir = data_dir

    def read_species_data(self):
        species_data = self.data_dir / "anonymised/species_data.parquet"
        if species_data.exists():
            return pd.read_parquet(species_data)
        else:
            raise FileNotFoundError("Anoymised species data not found, did you download/anonymise the data?")
    
    def read_plot_data(self):
        plot_data_file = self.data_dir / "anonymised/plot_data.gpkg"
        if plot_data_file.exists():
            plot_data = gpd.read_file(plot_data_file)
            # plot_data["geometry"] = gpd.points_from_xy(
            #     plot_data.Longitude, plot_data.Latitude, crs="EPSG:4326"
            # )
            # plot_gdf = gpd.GeoDataFrame(plot_data, geometry="geometry", crs="EPSG:4326")
            # # Convert date strings to datetime objects
            # plot_gdf["recording_date"] = pd.to_datetime(plot_gdf["recording_date"], format="%d.%m.%Y", errors='coerce')
            # plot_gdf = plot_gdf.drop(columns=["Date of recording"])
            return plot_data
        else:
            raise FileNotFoundError("Plot data not found, did you download/anonymise the data?")

    def load(self):
        # loading plot data
        plot_data = self.read_plot_data()
        # making dict from biodiv data
        species_data = self.read_species_data()
        return plot_data, species_data

if __name__ == "__main__":
    dataset = EVADataset()
    df_sp = dataset.read_species_data()
    dataset.load()
