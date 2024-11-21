from src.data_processing.utils_eva import EVADataset
from src.data_processing.utils_env_pred import CHELSADataset
from src.generate_SAR_data_EVA import read_CLM5_data, compile_SAR, compile_SAR_gpu, compile_SEAM
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
# %load_ext cudf.pandas  # pandas operations now use the GPU!
ENV_VARS = [
    "bio1", "pet_penman_mean", "sfcWind_mean", "bio4",
    "rsds_1981-2010_range_V.2.1", "bio12", "bio15"
]

def test_EVADataset():
    dataset = EVADataset()
    biodiv_df = dataset.read_biodiv_data()
    plot_df = dataset.read_plot_data()
    assert len(biodiv_df) > 0
    assert len(plot_df) > 0


def test_compile_SEAM():
    plot_gdf, dict_sp = EVADataset().load()
    SAR_data = compile_SAR(plot_gdf, dict_sp, 1000, (0.01, 0.01, 1, 1))
    assert all(SAR_data["sr"] > 1)
    
    chelsa = CHELSADataset().load().sel(variable=ENV_VARS)
    SEAM_df = compile_SEAM(SAR_data, chelsa)
    assert SEAM_df.loc[0, "env_pred"].shape[0] == len(ENV_VARS)

def test_compile_SEAM_gpu():
    plot_gdf, dict_sp = EVADataset().load()
    batchsize = 20
    poly_range = (0.01, 0.01, 1, 1)
    SAR_data = compile_SAR_gpu(plot_gdf, dict_sp, 1000, poly_range, batchsize)
    assert all(SAR_data["sr"] > 1)
    
    chelsa = CHELSADataset().load().sel(variable=ENV_VARS)
    SEAM_df = compile_SEAM(SAR_data, chelsa)
    assert SEAM_df.loc[0, "env_pred"].shape[0] == len(ENV_VARS)


        



dfa = []
for hab in HABITATS_EVA:
    data_dir = Path(f"../../../data/data_31_03_2023/{hab}")
    clm5_df = read_CLM5_data(data_dir)
    dfa.append(clm5_df)

df = pd.concat(dfa, ignore_index=True)
# make geopandas
df["geometry"] = gpd.points_from_xy(df.Longitude, df.Latitude, crs="EPSG:4326")
host_dataframe = gpd.GeoDataFrame(df, geometry="geometry")
host_dataframe.drop(["Latitude", "Longitude"], axis=1)

# for each habitat

# select a data point

# Make a random polygon based on this point

# select datapoints that intersect

# calculate unique species

# save polygon, together with SR and CLM5 data in same format as GBIF_polygon_CHELSA