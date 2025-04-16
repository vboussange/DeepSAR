import geopandas as gpd
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import random
from shapely.geometry import Polygon, Point, box
import numpy as np
import os
from pathlib import Path
import xarray as xr
import rioxarray
from rasterio.windows import from_bounds
from scipy import stats
from pathlib import Path

FILE_PATH = Path(__file__).parent


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

# 🔍 Example usage
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

class EUNISDataset():
    """
    Class to handle the EUNIS dataset specifically.
    """
    def __init__(self):
        legend_path = FILE_PATH / "../../data/raw/EUNIS/eunis_lev3_legend.txt"
        raster_path = FILE_PATH / "../../data/raw/EUNIS/eunis_map_lev3_full.tif"
        self.legend = self.load_legend(legend_path)
        self.raster = self.load_landcover(raster_path)
        # Assert that all raster values are found in the legend
        raster_values = np.unique(self.raster.values)
        legend_codes = self.legend["code"].values
        missing_values = [value for value in raster_values if value not in legend_codes and value != -1]
        assert not missing_values, f"Raster contains values not found in legend: {missing_values}"

    def load_legend(self, legend_path):
        """
        Load the EUNIS legend from a CSV file.
        """
        legend_df = pd.read_csv(legend_path, sep="\t", header=0, dtype={"code": int})
        legend_df["level_1"] = legend_df["name"].apply(extract_habitat_lev1)
        # return {
        #     int(k[0]): k[1][0:2]
        #     for k in zip(
        #         legend_df["code"],
        #         legend_df["name"],
        #     )
        # }
        return legend_df

    def load_landcover(self, raster_path):
        """
        Load and reproject the raster data to EPSG:3035.
        """
        # fixme: mask_and_scale=True works with rioxarray 15.0 but not 15.1
        with rioxarray.open_rasterio(raster_path,) as da:
            band1 = da.sel(band=1)
            lc_raster = band1.drop("band").rio.reproject("EPSG:3035")
            lc_raster = lc_raster.rename("EUNIS")
            return lc_raster.astype(int)
        
    def get_habitat_map(self, hab):
        """
        Get the habitat map for a specific habitat level. 1 for the habitat, 0 for other habitats, -1 for no data.
        """
        if hab not in self.legend["level_1"].unique():
            raise ValueError(f"Habitat level {hab} not found in legend.")
        habitat_codes = self.legend[self.legend["level_1"] == hab]["code"]
        habitat_map = xr.where(self.raster.isin(habitat_codes), 1, 0)
        habitat_map = habitat_map.where(self.raster > -1, -1)
        habitat_map = habitat_map.astype(int)
        return habitat_map
    
def get_fraction_habitat_landcover(habitat_map):
    numerator = (habitat_map == 1).sum()
    denominator = numerator + (habitat_map == 0).sum()
    return np.nan if denominator == 0 else float(numerator) / float(denominator)
        
        
if __name__ == "__main__":
    # Example usage
    eunis = EUNISDataset()
    print(eunis.legend)
    print(eunis.raster)
    
    # Plot the raster data
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    eunis.raster.plot(ax=ax, cmap='viridis', add_colorbar=True)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    plt.show()