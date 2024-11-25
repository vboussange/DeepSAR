"""
TODO: this script is not working yet. It should plot the locations of the EVA plots for each habitat.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
import matplotlib.pyplot as plt

from src.data_processing.utils_eva import EVADataset

CONFIG = {
    "crs": "EPSG:3035",
    "habitats": ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3"]
}

def load_and_preprocess_data():
    """
    Load and preprocess EVA data.
    Returns processed EVA dataset.
    """
    logging.info("Loading EVA data...")
    plot_gdf, _ = EVADataset().load()
    
    logging.info(f"Reprojecting to {CONFIG['crs']}...")
    plot_gdf = plot_gdf.to_crs(CONFIG["crs"])
    
    return plot_gdf

def plot_habitat_locations(plot_gdf):
    """
    Plot raw plot locations for each habitat.
    """
    plot_gdf_by_hab = plot_gdf.groupby("Level_2")
    
    for hab in CONFIG["habitats"]:
        logging.info(f"Plotting locations for habitat {hab}...")
        gdf_hab = plot_gdf_by_hab.get_group(hab)
        
        fig, ax = plt.subplots()
        gdf_hab.plot(ax=ax, marker='o', color='blue', markersize=5, label=hab)
        ax.set_title(f"Plot Locations for Habitat {hab}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    plot_gdf = load_and_preprocess_data()
    plot_habitat_locations(plot_gdf)
