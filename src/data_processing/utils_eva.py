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
EVA_CACHE = EVA_DATA_DIR / "cache.pkl"
COUNTRY_DATA = Path(__file__).parent / "../../data/NaturalEarth/ne_10m_admin_0_countries.shp"

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

class EVADataset:
    def __init__(self, data_dir=EVA_DATA_DIR, cache=EVA_CACHE):
        self.data_dir = data_dir
        self.cache = cache

    def read_species_data(self):
        species_data = self.data_dir / "anonymised/species_data.parquet"
        if species_data.exists():
            return pd.read_parquet(species_data)
        else:
            raise FileNotFoundError("Anoymised species data not found, did you download/anonymise the data?")
    
    def read_plot_data(self):
        plot_data_file = self.data_dir / "anonymised/plot_data.parquet"
        if plot_data_file.exists():
            return pd.read_parquet(plot_data_file)
        else:
            raise FileNotFoundError("Plot data not found, did you download/anonymise the data?")

    def clean_eva_plots(self, plot_gdf, dict_sp):
        # calculate SR per plot
        print("Discarding duplicates")
        plot_gdf["SR"] = [len(dict_sp[idx]) for idx in plot_gdf.index]
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

    def load(self):
        if not self.cache.exists():
            # loading plot data
            plot_data = self.read_plot_data()
            plot_data["geometry"] = gpd.points_from_xy(
                plot_data.Longitude, plot_data.Latitude, crs="EPSG:4326"
            )
            plot_gdf = gpd.GeoDataFrame(plot_data, geometry="geometry", crs="EPSG:4326")
            # Convert date strings to datetime objects
            plot_gdf["recording_date"] = pd.to_datetime(plot_gdf["recording_date"], format="%d.%m.%Y", errors='coerce')
            # plot_gdf = plot_gdf.drop(columns=["Date of recording"])

            # making dict from biodiv data
            biodiv_df = self.read_species_data()
            biodiv_gdf = biodiv_df.groupby("plot_id")
            dict_sp = {}
            for k, df in biodiv_gdf:
                dict_sp[k] = df.species.unique()

            # cleaning plot data
            plot_gdf = self.clean_eva_plots(plot_gdf, dict_sp)
            plot_gdf.drop(
                ["Latitude", "Longitude"], axis=1, inplace=True
            )
            
            # saving cache
            with open(self.cache, "wb") as pickle_file:
                cached_data = pickle.dump(
                    {"plot_gdf": plot_gdf, "dict_sp": dict_sp}, pickle_file
                )

            return plot_gdf, dict_sp
        else:
            with open(self.cache, "rb") as pickle_file:
                cached_data = pickle.load(pickle_file)
            return cached_data["plot_gdf"], cached_data["dict_sp"]

if __name__ == "__main__":
    dataset = EVADataset()
    df_sp = dataset.read_species_data()
    dataset.load()
