"""
Landcover data utilites
"""

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

EXTENT_DATASET = (
    -10.624685732460263,
    34.562189098269386,
    34.587857096739995,
    71.17876083332953,
)


class LandCoverDataset:

    def __init__(self, shape_files, raster_files, land_values):
        self.shape_files = shape_files
        self.raster_files = raster_files
        self.no_land_values = land_values

        # making sure that repository already exist
        for d in [shape_files, raster_files]:
            for f in d.values():
                f.parent.mkdir(parents=True, exist_ok=True)

    def load_shapefile(self, name, **kwargs):
        """
        Method to load the vectorized dataset.
        """

        shape_file = self.shape_files[name]

        gdf = gpd.read_parquet(
            shape_file, **kwargs
        )  # see https://autogis-site.readthedocs.io/en/2021/notebooks/L2/02-projections.html
        return gdf

    def get_land_support(self, geo_bounds, name="original"):
        with rioxarray.open_rasterio(
            self.raster_files[name], mask_and_scale=True
        ) as da:
            band1 = da.sel(band=1)
            band1_eu = band1.rio.isel_window(
                from_bounds(*geo_bounds, transform=da.rio.transform())
            )
            land_support = np.logical_or(
                band1_eu == self.no_land_values[0], band1_eu == self.no_land_values[1]
            )
            for v in self.no_land_values[2:]:
                land_support = np.logical_or(land_support, band1_eu == v)
            land_support = ~land_support
            return land_support.astype(np.int16)

    def is_land(self, v):
        return ~(self.no_land_values == v).any()


class LandSysDataset(LandCoverDataset):
    """
    10.1007/s10980-021-01227-5
    """

    def __init__(self):
        no_land_values = np.array([0, 11, 13, 21, 22, 23, 31, 32, 61, 62, 63])
        raster_files = {
            "original": Path(FILE_PATH, "../../../data/landsys/EU_landSystem.tif")
        }
        shape_files = {
            "c4": Path(FILE_PATH, "../../../data/landsys/EU_landSystem_4.parquet"),
            "c8": Path(FILE_PATH, "../../../data/landsys/EU_landSystem_8.parquet"),
        }
        legend_file = Path(FILE_PATH, "../../../data/landsys/legend.csv")
        LandCoverDataset.__init__(self, shape_files, raster_files, no_land_values)
        self.legend_file = legend_file

    def get_habitat_id_from_habitat_name(self, name):
        legend = pd.read_csv(self.legend_file)
        return legend[legend.Description == name].Code.iloc[0]

    def get_habitat_name_from_habitat_ID(self, name):
        legend = pd.read_csv(self.legend_file)
        return legend[legend.Code == name].Description.iloc[0]

    def load_landcover_level3(self):
        """
        Loads level 3 habitat type, i.e. original habitat definition
        """
        with rioxarray.open_rasterio(
            self.raster_files["original"], mask_and_scale=True
        ) as da:
            band1 = da.sel(band=1)
            habitat_id0 = band1.drop("band")
            habitat_id0 = habitat_id0.rename("LandSysDatasetL3")
            return habitat_id0

    def load_landcover_level1(self):
        """
        Loads level 1 habitat type.

        ## Details
        Converting level 3 into level 1 (open vs closed).
        Converts seas and lakes and glaciers to 0, to distinguish from wetlands (id: 80)
        """
        with rioxarray.open_rasterio(
            self.raster_files["l1"], mask_and_scale=True
        ) as da:
            band1 = da.sel(band=1)
            habitat_id0 = band1.drop("band")
            # need to convert to integer before str
            habitat_id0 = habitat_id0.astype(np.int16)
            # transforming water bodies and glacier to 0 (na), so as not to mistake them with wetlands
            habitat_id0.values[habitat_id0 == 11] = 0
            habitat_id0.values[habitat_id0 == 13] = 0
            habitat_id0 = habitat_id0.astype(str)
            # selecting two first digits
            habitat_id0.values = np.vectorize(lambda x: x[:1])(habitat_id0.values)
            # back to integers
            habitat_id0 = habitat_id0.astype(np.int16)
            habitat_id0 = habitat_id0.rename("LandSysDatasetL1")
            return habitat_id0


# TODO: to be completed
# class EUNISDataset(LandCoverDataset):
#     """
#     https://www.eea.europa.eu/en/datahub/datahubitem-view/adbb2781-2d4d-4a6c-8ce1-875bae0f6703
#     """
#     TIF_FILE_LANDSYS = Path(FILE_PATH, "../../../data/landsys/EU_landSystem.tif")
#     SHAPE_FILE_LANDSYS_4 = Path(FILE_PATH,
#                                 "../../../data/landsys/EU_landSystem_4.parquet")
#     SHAPE_FILE_LANDSYS_8 = Path(FILE_PATH,
#                                 "../../../data/landsys/EU_landSystem_8.parquet")
#     LEGEND_FILE_LANDSYS = Path(FILE_PATH, "../../../data/landsys/legend.csv")

#     def __init__(self):
#         NO_LAND_VALUES = np.array([0, 11, 13, 21, 22, 23, 31, 32, 61, 62, 63])
#         LandCoverDataset.__init__(self,
#                                   SHAPE_FILE_LANDSYS_4,
#                                   SHAPE_FILE_LANDSYS_8,
#                                   TIF_FILE_LANDSYS,
#                                   NO_LAND_VALUES)
#         self.legend_file = LEGEND_FILE_LANDSYS

#     def get_habitat_id_from_habitat_name(self, name):
#         legend = pd.read_csv(self.legend_file)
#         return legend[legend.Description == name].Code.iloc[0]

#     def get_habitat_name_from_habitat_ID(self, name):
#         legend = pd.read_csv(self.legend_file)
#         return legend[legend.Code == name].Description.iloc[0]

#     def load_landcover_level3(self):
#         """
#         Loads level 3 habitat type, i.e. original habitat definition
#         """
#         with rioxarray.open_rasterio(self.raster_file,
#                                      mask_and_scale=True) as da:
#             band1 = da.sel(band=1)
#             habitat_id0 = band1.drop("band")
#             habitat_id0 = habitat_id0.rename("LandSysDatasetL3")
#             return habitat_id0

#     def load_landcover_level1(self):
#         """
#         Loads level 1 habitat type.

#         ## Details
#         Converting level 3 into level 1 (open vs closed).
#         Converts seas and lakes and glaciers to 0, to distinguish from wetlands (id: 80)
#         """
#         with rioxarray.open_rasterio(self.raster_file,
#                                      mask_and_scale=True) as da:
#             band1 = da.sel(band=1)
#             habitat_id0 = band1.drop("band")
#             # need to convert to integer before str
#             habitat_id0 = habitat_id0.astype(np.int16)
#             # transforming water bodies and glacier to 0 (na), so as not to mistake them with wetlands
#             habitat_id0.values[habitat_id0 == 11] = 0
#             habitat_id0.values[habitat_id0 == 13] = 0
#             habitat_id0 = habitat_id0.astype(str)
#             # selecting two first digits
#             habitat_id0.values = np.vectorize(lambda x: x[:1])(
#                 habitat_id0.values)
#             # back to integers
#             habitat_id0 = habitat_id0.astype(np.int16)
#             habitat_id0 = habitat_id0.rename("LandSysDatasetL1")
#             return habitat_id0


class CopernicusDataset(LandCoverDataset):
    """
    10.5281/zenodo.3939050
    """

    # 200: seas, oceans, transformed to 80
    # 80 : lakes, transformed into waterbodies
    # 70 snow and ice
    # 50 urban / built up
    # 40 cropland
    # 0 no input data available

    tif_files = {
        "original": Path(
            "/lud11/boussang/data/copernicus_landcover/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
        ),
        "l2": Path(
            "/lud11/boussang/data/copernicus_landcover/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326_habitat_id0.nc"
        ),
        "l3_1km": Path(
            "/lud11/boussang/data/copernicus_landcover/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326_1km.nc"
        ),
    }
    shape_files = {
        "c4": Path(
            FILE_PATH, "../../../data/copernicus_landcover/EU_COPERNICUS_LC_4.parquet"
        ),
        "c8": Path(
            FILE_PATH, "../../../data/copernicus_landcover/EU_COPERNICUS_LC_8.parquet"
        ),
        "c8_1km": Path(
            FILE_PATH,
            "../../../data/copernicus_landcover/EU_COPERNICUS_LC_8_l3_1km.parquet",
        ),
    }

    def __init__(self, shape_files=shape_files, tif_files=tif_files):

        # notice that this legend is also included in the attributes of the tif file
        legend_l2 = {
            0: "No data",
            11: "Closed forest",
            12: "Open forest",
            20: "Shrubs",
            30: "Herbaceous vegetation",
            90: "Herbaceous wetlands",
            10: "Moss and lichen",
            60: "Bare / sparse vegetation",
        }
        no_land_values = np.array([200, 80, 70, 50, 40, 0])
        LandCoverDataset.__init__(self, shape_files, tif_files, no_land_values)
        self.legend_l2 = legend_l2
        self.legend_l3 = None

    def load_landcover_level2(self, extent=EXTENT_DATASET):
        """
        Loads level 2 habitat type, along extent defined by global variable `EXTENT_DATASET`.

        ## Details
        Converting level 3 forest habitat types into level 2 (open vs closed).
        Converts seas and lakes to water bodies (id: 80)
        Caches to netcdf file.
        """
        if not self.raster_files["l2"].is_file():
            with rioxarray.open_rasterio(
                self.raster_files["l2"], mask_and_scale=True
            ) as da:
                band1 = da.sel(band=1).rio.clip_box(*extent)
                habitat_id0 = band1.drop("band")
                # need to convert to integer before str
                habitat_id0 = habitat_id0.astype(np.int16)
                # transforming ocean habitat type to 80 (water bodies), so that we do not mistake them for shrubs (20)
                habitat_id0.values[habitat_id0 == 200] = 80
                habitat_id0 = habitat_id0.astype(str)
                # selecting two first digits
                habitat_id0.values = np.vectorize(lambda x: x[:2])(habitat_id0.values)
                # back to integers
                habitat_id0 = habitat_id0.astype(np.int16)
                ds = habitat_id0.to_dataset(name="habitat_id0")
                ds.to_netcdf(self.raster_files["l2"])
                habitat_id0 = habitat_id0.rename("CopernicusL2")
                return habitat_id0
        with xr.open_dataset(self.raster_files["l2"]) as ds:
            return ds["habitat_id0"].rio.clip_box(*extent)

    def add_legend_l3(self, raster):
        self.legend_l3 = {
            int(k[0]): k[1]
            for k in zip(
                raster.attrs["flag_values"].split(","),
                raster.attrs["flag_meanings"].split(","),
            )
        }

    def load_landcover_level3(self, extent=EXTENT_DATASET):
        """
        Loads level 3 habitat type, i.e. original habitat definition
        """
        with rioxarray.open_rasterio(
            self.raster_files["original"], mask_and_scale=True
        ) as da:
            band1 = da.sel(band=1).rio.clip_box(*extent)
            habitat_id0 = band1.drop("band")
            self.add_legend_l3(habitat_id0)
            habitat_id0 = habitat_id0.rename("CopernicusL3")
            return habitat_id0.astype(np.int16)

    def load_landcover_level3_1km(self, extent=EXTENT_DATASET):
        """
        Loads level 3 habitat type, i.e. original habitat definition, down-sampled at 1km
        """
        if not self.raster_files["l3_1km"].is_file():
            with rioxarray.open_rasterio(
                self.raster_files["l3_1km"], mask_and_scale=True
            ) as da:
                band1 = da.sel(band=1).rio.clip_box(*extent)
                habitat_id = band1.drop("band")

                def calc_mode(arr, axis):
                    mode_val = stats.mode(arr, axis=axis, nan_policy="omit")
                    return mode_val[0]

                coarsened = habitat_id.coarsen(x=10, y=10, boundary="trim")
                mode_data = coarsened.reduce(calc_mode)
                raster_1km = mode_data.astype(np.int16)
                ds = raster_1km.to_dataset(name="raster_1km")
                ds.to_netcdf(self.raster_files["l3_1km"])

                self.add_legend_l3(raster_1km)
                habitat_id0 = habitat_id0.rename("Copernicus1KML3")

                return raster_1km

        with xr.open_dataset(self.raster_files["l3_1km"]) as ds:
            raster = ds["raster_1km"].rio.clip_box(*extent)
            self.add_legend_l3(raster)
            raster = raster.rename("Copernicus1KML3")

            return raster


class EUNISDataset(LandCoverDataset):
    """
    Class to handle the EUNIS dataset specifically.
    """
    raster_files = {"original" : Path(
            FILE_PATH, "../../../data/EUNIS_raw/sara/eunis_map_current.tif"
        )}
    legend_path = Path(
            FILE_PATH, "../../../data/EUNIS_raw/sara/eunis_legend.csv"
        )
    def __init__(self, raster_files=raster_files, legend_path=legend_path, shape_files={}):
        super().__init__(shape_files, raster_files, np.array([]))  # No land values specified
        self.legend_path = legend_path
        self.legend = self.load_legend()

    def load_legend(self):
        """
        Load the EUNIS legend from a CSV file.
        """
        legend_df = pd.read_csv(self.legend_path, usecols=["mapid", "EUNIS_2020_code"])
        return {
            int(k[0]): k[1][0:2]
            for k in zip(
                legend_df["mapid"],
                legend_df["EUNIS_2020_code"],
            )
        }

    def load_landcover(self):
        """
        Load and reproject the raster data to EPSG:4326.
        """
        # fixme: mask_and_scale=True works with rioxarray 15.0 but not 15.1
        with rioxarray.open_rasterio(self.raster_files["original"], mask_and_scale=True) as da:
            band1 = da.sel(band=1)
            lc_raster = band1.drop("band").rio.reproject("EPSG:4326")
            lc_raster = lc_raster.rename("EUNIS-Sara")
            lc_raster = lc_raster.fillna(-1)
            return lc_raster.astype(int)



def crs_transform_and_area(gdf):
    print(
        "Calculating areas"
    )  # see https://stackoverflow.com/questions/72073417/userwarning-geometry-is-in-a-geographic-crs-results-from-buffer-are-likely-i
    # but see also EPSG:3035 (see https://gis.stackexchange.com/questions/182417/what-is-the-proj4string-of-the-lambert-azimuthal-equal-area-projection)
    gdf["area"] = gdf.to_crs(crs=3035).geometry.area
    print("Changing crs...")
    gdf = gdf.to_crs(epsg=4326)
    return gdf
