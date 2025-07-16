import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from pathlib import Path
from math import radians
import geopandas as gpd
from tqdm import tqdm 
from deepsar.data_processing.utils_polygons import random_size_from_range, place_randomly_rectangle
from shapely.geometry import Point, MultiPoint, box
from scipy.spatial import KDTree
import logging 
import warnings

def batch_indices(N, batch_size):
    """Yield successive batch-sized chunks of indices from 0 to N."""
    for i in range(0, N, batch_size):
        yield range(i, min(i + batch_size, N))

# working with dictionnary of species
def clip_EVA_SR(plot_gdf, species_data, polygons_gdf, verbose=False):
    data = pd.DataFrame({
        "area": pd.Series(int),
        "sr": pd.Series(int),
        "num_plots": pd.Series(int),
    })
    for i, poly in tqdm(enumerate(polygons_gdf.geometry), desc="Clipping SR", total=len(polygons_gdf), disable=not verbose):
        df_samp = plot_gdf[plot_gdf.within(poly)]
        species = np.concatenate([species_data[idx] for idx in df_samp.index])
        sr = len(np.unique(species))
        a = np.sum(df_samp['area'])
        # geom = MultiPoint(df_samp.geometry.to_list())
        num_plots = len(df_samp)
        data.loc[i, ["area", "sr", "num_plots"]] = [a, sr, num_plots]
    return data


def clip_EVA_SR_gpu(plot_gdf, dict_sp, polygons_gdf, batchsize):
    try:
        import cuspatial
    except ImportError:
        raise ImportError("This function requires cuspatial for GPU acceleration. Please install cuspatial and try again.")
   
    data = pd.DataFrame({"area" : pd.Series(np.int16), "sr": pd.Series(np.int16), "geometry" : pd.Series("object"), "plot_idxs" : pd.Series("object")})
    
    # sending to gpu
    logging.debug("Sending gbif_data to GPU")
    plot_gdf_gpu = cuspatial.from_geopandas(plot_gdf)
    logging.debug("Sending polygons to GPU")
    polygon_gdf_gpu = cuspatial.from_geopandas(polygons_gdf.geometry)
    
    gen = batch_indices(len(polygons_gdf), batchsize)
    for batch in tqdm(gen):
        idxs = list(batch)
        pip = cuspatial.point_in_polygon(plot_gdf_gpu.geometry, polygon_gdf_gpu.iloc[idxs])
        
        for (i,col) in enumerate(pip.columns):
            plot_idxs = plot_gdf_gpu.index[pip[col]].to_numpy()
            species = np.concatenate([dict_sp[idx] for idx in plot_idxs])
            sr = len(np.unique(species))
            a = plot_gdf_gpu[pip[col]]['plot_size'].sum()
            geom = MultiPoint(plot_gdf.geometry[plot_idxs].to_list())
            data.loc[idxs[i], ["area", "sr", "geometry", "plot_idxs"]] = [a, sr, geom, plot_idxs]

    return gpd.GeoDataFrame(data, crs = plot_gdf.crs, geometry="geometry")

    
def generate_random_square(candidate_points, area_range):
    """
    Generates a random square with an area sampled from a given range. 
    """
    log_area_range = np.log(area_range)
    point = np.random.choice(candidate_points.geometry)
    # Sample log_area with probability inversely proportional to log_area
    # Use inverse transform sampling for p(x) ∝ 1/x over [a, b]
    # TODO: to check if this is correct
    a, b = log_area_range
    u = np.random.uniform(0, 1)
    log_area = a * (b / a) ** u
    log_length = log_area / 2
    height = np.exp(log_length)
    length = np.exp(log_length)
    
    # Calculate half dimensions to center the box around the point
    half_length = length / 2
    half_height = height / 2
    
    # Create the box centered on the point
    new_box = box(
        point.x - half_length, 
        point.y - half_height, 
        point.x + half_length, 
        point.y + half_height
    )
    
    return new_box