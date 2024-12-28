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


DATA_DIR = Path(__file__).parent / "../../data/EVA"
EVA_CACHE = DATA_DIR / "cache.pkl"
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
    def __init__(self, data_dir=DATA_DIR, cache=EVA_CACHE):
        self.data_dir = data_dir
        self.cache = cache

    def read_biodiv_data(self):
        parquet_file = self.data_dir / "biodiv_df.parquet"
        if parquet_file.exists():
            return pd.read_parquet(parquet_file)
        else:
            bio_df = pd.read_csv(
                Path(self.data_dir) / "vpl_all.csv",
                header=0,
                usecols=["plot_id", "species"],
                sep=",",
                engine="python",
            )
            bio_df.to_parquet(parquet_file)
            return bio_df

    def read_plot_data(self):
        parquet_file = self.data_dir / "plot_df.parquet"
        if parquet_file.exists():
            return pd.read_parquet(parquet_file)
        else:
            bio_df = pd.read_csv(
                Path(self.data_dir) / "hea_all.csv",
                header=0,
                usecols=[
                    "plot_id",
                    "Level_2",
                    "Level_2_name",
                    "Longitude",
                    "Latitude",
                    "plot_size",
                    "uncertainty_m",
                ],
                sep=",",
                engine="python",
                index_col="plot_id",
            )
            bio_df.to_parquet(parquet_file)
            return bio_df

    def clean_eva_plots(self, plot_gdf, dict_sp):
        # calculate SR per plot
        print("Discarding duplicates")
        plot_gdf["SR"] = [len(dict_sp[idx]) for idx in plot_gdf.index]
        # identify unique locations and plot_id with highest SR
        plot_idx = []
        for _, _gdf in plot_gdf.groupby("geometry"):
            plot_idx.append(_gdf["SR"].idxmax())
        # keep only plots with high SR
        plot_gdf = plot_gdf.loc[plot_idx]

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
        plot_gdf = plot_gdf[plot_gdf.uncertainty_m < 1000]

        return plot_gdf

    def load(self):
        if not self.cache.exists():
            # loading plot data
            plot_df = self.read_plot_data()
            plot_df["geometry"] = gpd.points_from_xy(
                plot_df.Longitude, plot_df.Latitude, crs="EPSG:4326"
            )
            plot_gdf = gpd.GeoDataFrame(plot_df, geometry="geometry", crs="EPSG:4326")

            # making dict from biodiv data
            biodiv_df = self.read_biodiv_data()
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
    # trying to create a wide table format.
    # takes for ever
    dataset = EVADataset()
    df_sp = dataset.read_biodiv_data()
    df_sp['presence'] = True
    
    # pivoting, this takes some time
    wide_df_sp = df_sp.pivot(index='plot_id', columns='species', values='presence')
    wide_df_sp = wide_df_sp.fillna(False) # takes for ever
    
    
    size_in_mb = wide_df_sp.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Size of dataframe: {size_in_mb:.2f} MB")

    wide_df_sp.to_parquet("wide_df_vpl_all.parquet", engine='pyarrow', index=True)
