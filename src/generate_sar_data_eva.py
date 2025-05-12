import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from pathlib import Path
from math import radians
import geopandas as gpd
from tqdm import tqdm 
from src.data_processing.utils_polygons import random_size_from_range, place_randomly_rectangle
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


# attempt to make it work with a dataframe of species
# wide format would be the most relevant, but takes too much memory
# def clip_GBIF_SR(plot_gdf, sp_df, polygons_gdf):
#     assert plot_gdf.crs == polygons_gdf.crs, "GeoPandas must be in same CRS"
#     plot_gdf = plot_gdf.reset_index(drop=True)
#     polygons_gdf = polygons_gdf.reset_index(drop=True)
#     polygons_gdf["sr"] = np.int16(0)

#     for i, row in polygons_gdf.iterrows():
#         pip = plot_gdf.within(row.geometry)
#         species = sp_df.speciesKey[pip].to_numpy()
#         sr = len(np.unique(species))
#         polygons_gdf.loc[i, "sr"] = sr
#     return polygons_gdf
     
"""
Used for EVA

"""
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

def stats_min_pairwise_distance(multipoint):
    if len(multipoint.geoms) < 2:
        return 0, 0
    
    # Extract point coordinates from the MultiPoint object
    points = np.array([point.coords[0] for point in multipoint.geoms])
    
    # Build a k-d tree for efficient nearest-neighbor search
    tree = KDTree(points)
    
    # Query the nearest neighbor for each point (k=2 because the nearest is the point itself)
    distances, _ = tree.query(points, k=2)
    
    # distances[:, 1] contains the distance to the nearest neighbor (excluding itself)
    # Calculate the average of these distances
    average_distance = np.mean(distances[:, 1])
    std_distance = np.std(distances[:, 1])
    
    return average_distance, std_distance

def generate_random_boxes_from_candidate_pairs(candidate_points, num_boxes):
    """
    Generates random rectangular boxes by picking two candidate points from candidate_points.
    Each box is the axis‐aligned rectangle defined by the two points.
    
    Parameters:
    - candidate_points: A GeoDataFrame containing candidate points with a 'geometry'
      column and a 'partition' column.
    - num_boxes: The number of boxes to generate.
    
    Returns:
    - A GeoDataFrame containing the generated boxes with the same CRS as candidate_points.
    """
    boxes = []
    
    for _ in range(num_boxes):
        # Pick two distinct candidate points at random.
        idxs = np.random.choice(candidate_points.index, 2, replace=False)
        point1 = candidate_points.geometry[idxs[0]]
        point2 = candidate_points.geometry[idxs[1]]
        
        # Determine the lower left and upper right coordinates.
        min_x = min(point1.x, point2.x)
        min_y = min(point1.y, point2.y)
        max_x = max(point1.x, point2.x)
        max_y = max(point1.y, point2.y)
        
        # Create the box.
        # We add and subtract 1 to the coordinates to ensure that the box contains the points.
        new_box = box(min_x - 1, min_y - 1, max_x + 1, max_y + 1)
        boxes.append(new_box)
        
    
    return gpd.GeoDataFrame(
        geometry=boxes,
        crs=candidate_points.crs
    )
    
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

def generate_random_squares(candidate_points, num_boxes, area_range):
    """
    Generates a specified number of random squares a given
    area.
    The generated squares are centered on points randomly sampled from `candidate_points`.
    
    TODO: While the box is centered on the candidate point, parts of it may fall outside bounds.
    TODO: The box should be cropped given a certain bounding box (to be added as an argument).

    Returns:
    - A GeoDataFrame containing the generated boxes with the same CRS
    as `candidate_points`.
    """

    boxes = []
    log_area_range = np.log(area_range)
    for idx in np.random.choice(candidate_points.index, num_boxes):
        point = candidate_points.geometry[idx]
        log_area = np.random.uniform(*log_area_range)
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
        boxes.append(new_box)

    return gpd.GeoDataFrame(geometry=boxes, crs=candidate_points.crs)


def generate_random_boxes(candidate_points, num_boxes, area_range, side_range):
    """
    Generates a specified number of random rectangular boxes within a given
    area, ensuring that each box's area and side lengths fall within specified
    ranges.
    The generated rectangles are centered on points randomly sampled from `candidate_points`.
    
    TODO: While the box is centered on the candidate point, parts of it may fall outside bounds.
    TODO: The box should be cropped given a certain bounding box (to be added as an argument).

    Parameters:
    - candidate_points: A GeoDataFrame containing points that will
    serve as the centers for the generated boxes.
    - num_boxes: The number of boxes to generate.
    - area_range: A tuple specifying the minimum and maximum area
    for each box, in same units than geometry of `candidate_points`
    - side_range: A tuple specifying the minimum and maximum side
    length for each box, in same units than geometry of `candidate_points`.

    Returns:
    - A GeoDataFrame containing the generated boxes with the same CRS
    as `candidate_points`.
    """

    boxes = []
    log_area_range = np.log(area_range)
    log_side_range = np.log(side_range)
    for idx in np.random.choice(candidate_points.index, num_boxes):
        point = candidate_points.geometry[idx]
        log_length = np.random.uniform(*log_side_range)
        log_height_min = max(
            log_side_range[0], log_area_range[0] - log_length
        )  # log of a fraction is the difference of the log of the numerator and the log of the denominator
        log_height_max = min(log_side_range[1], log_area_range[1] - log_length)
        log_height = np.random.uniform(log_height_min, log_height_max)
        height = np.exp(log_height)
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
        boxes.append(new_box)

    return gpd.GeoDataFrame(geometry=boxes, crs=candidate_points.crs)


def crop_raster(lc_binary, geom):
    assert lc_binary.rio.crs == "EPSG:3035"
    minx, miny, maxx, maxy = geom.bounds
    cropped_lcb = lc_binary.sel(x=slice(minx, maxx), y=slice(miny, maxy))
    return cropped_lcb