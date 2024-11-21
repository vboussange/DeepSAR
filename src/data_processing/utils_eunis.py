"""
Script handling EUNIS data


# question to Dirk: what is the difference between ID, value and name
# question to Dirk: what's the CRS used?
# question to Dirk: what do multipolygons correspond to?
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

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
SHAPEFILE_PATH_COUNTRY = os.path.join(FILE_PATH,'../../../data/CH_shapefile/ne_10m_admin_0_countries_lakes/ne_10m_admin_0_countries_lakes.shp')
CACHE_PATH = os.path.join(FILE_PATH,'../../../data/.cache_eunis/')
SHAPE_FILE_EUNIS = os.path.join(FILE_PATH,'../../../data/eunis_l2/eunis_l2.shp')
LEGEND_FILE_EUNIS = os.path.join(FILE_PATH,'../../../data/eunis_l2/legend_eunis_l2.txt')
    
# Load the shapefile and legend
def load_eunis_data(**kwargs):
    gdf = gpd.read_file(SHAPE_FILE_EUNIS, **kwargs).to_crs(epsg=3035) # see https://autogis-site.readthedocs.io/en/2021/notebooks/L2/02-projections.html
    return gdf

def prepare_eunis_data(cache_path):
    p = Path(cache_path)
    p.parents[0].mkdir(parents=True, exist_ok=True)
    if not os.path.exists(cache_path):
        print("Loading raw EUNIS data...")
        gdf = load_eunis_data()
        gdf['area'] = gdf.geometry.area
        print("Changing crs...")
        gdf = gdf.to_crs(epsg=4326)
        print("Filtering MultiPolygons")
        n_before = len(gdf)
        gdf = gdf[gdf.geom_type=='Polygon']
        print(n_before-len(gdf), " points were discarded")
        print("caching...")
        gdf.to_parquet(cache_path)
        return gdf
    else:
        return gpd.read_parquet(cache_path)

# seems like `code` is shift by 1 compared to `value` column in shape file
def get_value_from_name_habitat(name):
    legend = pd.read_csv(LEGEND_FILE_EUNIS)
    return str(legend[legend.EUNIS_l2 == name].code.iloc[0] -1)

def add_feature(ax):
    # Add features for context
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    # ax.stock_img()

if __name__ == "__main__":
    
    gdf = load_eunis_data()

    # Display the first few rows of the GeoDataFrame
    print(gdf.head())

    # investigating polygons
    multipolygon_gdf = gdf[gdf.geometry.type == 'MultiPolygon']
    multipolygon_gdf
    multipolygon_gdf.shape
    multipolygon_gdf.geometry.iloc[-9]

    # investigating forest t_1
    habitat = 'T1'
    subg_gdf = gdf[gdf.NAME == get_value_from_name_habitat(habitat)]
    subg_gdf['area'] = subg_gdf.geometry.area
    
    # Plot all forest habitats
    crs_cpy = ccrs.epsg(3035)
    fig, ax = plt.subplots(subplot_kw={'projection': crs_cpy})
    add_feature(ax)
    subg_gdf.plot(ax=ax)
    ax.set_title('Forest Habitats')
    fig


    # Plot size distribution of habitats
    fig, ax = plt.subplots()
    subg_gdf['area'].plot(kind='hist', bins=40, ax=ax)
    ax.set_title('Size Distribution of Habitats')
    ax.set_xlabel('Area')
    ax.set_ylabel('Frequency')
    ax.set_yscale("log")
    # ax.set_xscale("log")
    fig
    
    # Selecting an area and a random point within
    sorted_gdf = subg_gdf.sort_values(by='area', ascending=False)
    poly = sorted_gdf.geometry.iloc[50]
    point = get_random_point_within_polygon(poly)
    
    # Plot the intersection between poly and a box with center the random point, on a map
    box_around_point = create_box_around_point(point, 2e4)
    intersection = poly.intersection(box_around_point)

    combined_geom = gpd.GeoSeries([poly, box_around_point, intersection], crs="EPSG:3035").unary_union
    minx, miny, maxx, maxy = combined_geom.bounds

    # Set a buffer around the combined bounds for better visualization
    buffer = 5e4  # This is a degree buffer; adjust as needed
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer

    # Plot the intersection with a map
    fig, ax = plt.subplots(subplot_kw={'projection': crs_cpy})

    # Set the extent to focus on the area of interest
    ax.set_extent([minx, maxx, miny, maxy], crs=crs_cpy)
    ax.add_geometries([poly], crs=crs_cpy, facecolor='blue', edgecolor='black', alpha=0.5)
    ax.add_geometries([box_around_point], crs=crs_cpy, facecolor='green', edgecolor='black', alpha=0.5)
    ax.add_geometries([intersection], crs=crs_cpy, facecolor='red', edgecolor='black', alpha=0.5)

    add_feature(ax)

    ax.set_title('Polygon, Box, and Intersection on Map')
    fig

    ## Plotting intersections with boxes of increasing area
    # Define a range of sizes for the boxes
    box_sizes = np.linspace(1e3, 1e5, 10)  # Adjust the range and number of sizes as needed

    # Generate intersections
    intersections = generate_intersections(poly, point, box_sizes)

    # Plot the intersections with a map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.epsg(3035)})

    # Set the extent to focus on the area of interest
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.epsg(3035))

    # Add features for context
    add_feature(ax)

    # Plot each intersection
    colors = plt.cm.cool(np.linspace(0, 1, len(intersections)))  # Color map
    for intersection, color in zip(intersections[::-1], colors):
        ax.add_geometries([intersection], crs=ccrs.epsg(3035), facecolor=color, alpha=1)

    ax.set_title('Intersections with Increasing Box Sizes on Map')
    plt.show()