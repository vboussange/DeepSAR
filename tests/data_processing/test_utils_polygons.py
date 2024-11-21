import os
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon, Point, MultiPoint
import cuspatial
import numpy as np
import rioxarray
import xarray as xr

from src.data_processing.utils_landcover import LandSysDataset
from src.data_processing.utils_gbif_local import process_partition
from src.data_processing.utils_polygons import (generate_random_polygons, 
                                                size_from_template, 
                                                random_size_from_range, 
                                                get_random_points_within_polygon, 
                                                get_random_points_within_polygon_gpu, 
                                                place_randomly_rectangle,
                                                lc_raster_to_multipoint)

os.chdir(Path(__file__).parent)


def test_generate_random_polygons_random_size_from_range():
    """
    These are more visual tests
    """

    NUM_POLYGONS = 1000
    lc_dataset = LandSysDataset()
    gdf_habitat_free = generate_random_polygons(NUM_POLYGONS, lc_dataset,
                                                random_size_from_range)
    gdf_habitat_free = gdf_habitat_free.to_crs(epsg=4326)
    #######################################
    #### Plotting #########################
    #######################################
    # Load world map for background
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    # Plot the world map
    gdf_habitat_free.plot(ax=ax)
    # world.boundary.plot(ax=ax, linewidth=1, color="black")


def test_generate_random_polygons_size_from_template():
    """
    These are more visual tests. results should ressemble `test_shapefile` outputs
    """
    # testing
    NUM_POLYGONS = 50000
    lc_dataset = LandSysDataset()
    gdf_habitat_free = generate_random_polygons(NUM_POLYGONS, lc_dataset,
                                                size_from_template)
    gdf_habitat_free = gdf_habitat_free.to_crs(epsg=4326)
    #######################################
    #### Plotting #########################
    #######################################
    # Load world map for background
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    # Plot the world map
    gdf_habitat_free.plot(ax=ax)
    # world.boundary.plot(ax=ax, linewidth=1, color="black")


def test_shapefile():
    NUM_POLYGONS = 10000
    lc_dataset = LandSysDataset()
    gdf_template = lc_dataset.load_shapefile(4)
    gdf = gdf_template.iloc[np.random.choice(range(len(gdf_template)),
                                             NUM_POLYGONS).tolist()]
    gdf.plot()


def test_point_in_polygon():
    """
    TODO: This code needs to be cleaned
    """
    # Load GBIF and EUNIS data
    print("Loading GBIF data...")
    gbif_gdf_cpu = load_gbif_data()
    gbif_gdf_gpu = cuspatial.from_geopandas(
        gbif_gdf_cpu[["gbifID", "dateIdentified", "geometry"]])

    print("Loading EUNIS data...")
    polygon_gdf = load_and_cache_eunis_data_for_country('T3')
    polygon_gdf_gpu = cuspatial.from_geopandas(polygon_gdf)
    polygon = polygon_gdf.geometry.iloc[9]

    n_gpu = cuspatial.point_in_polygon(
        gbif_gdf_gpu.geometry,
        polygon_gdf_gpu.geometry.iloc[9:10]).sum(axis=0).to_numpy()[0]
    n_cpu = gbif_gdf_cpu.geometry.within(polygon).sum()

    assert n_gpu == n_cpu


def test_get_random_points_within_polygon():
    p1 = Polygon([(0, 0), (1, 0), (1, 1)])
    npoints = 1000000
    points = get_random_points_within_polygon(npoints, p1)
    assert len(points) == npoints


def test_get_random_points_within_polygon_gpu():
    polygon = Polygon([(0, 0), (1, 0), (1, 1)])
    npoints = 100000
    points = get_random_points_within_polygon_gpu(npoints, polygon)
    assert len(points) == npoints


def test_generate_random_polygons_size_from_template():
    NUM_POLYGONS = 1000
    lc_dataset = LandSysDataset()
    gdf_habitat_free = generate_random_polygons(NUM_POLYGONS, lc_dataset,
                                                size_from_template)
    assert len(gdf_habitat_free) == NUM_POLYGONS


def test_generate_random_polygons_random_size_from_range():
    NUM_POLYGONS = 1000
    lc_dataset = LandSysDataset()
    gdf_habitat_free = generate_random_polygons(NUM_POLYGONS, lc_dataset,
                                                random_size_from_range)
    assert len(gdf_habitat_free) == NUM_POLYGONS


def test_process_partition():
    poly_range = (0.01, 0.01, 1, 1)
    geominx, geominy, geomaxx, geomaxy = (0, 0, 4, 4)
    batchsize = 20
    nspecies = 100
    num_polygons = batchsize

    # create random polygons_gdf
    polygons = []
    for _ in range(num_polygons):
        p = Point(np.random.uniform(geominx, geomaxx),
                  np.random.uniform(geominy, geomaxy))
        length, height = random_size_from_range(p, poly_range)
        polygons.append(place_randomly_rectangle(p, length, height))

    polygons_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")
    polygons_gdf["area"] = polygons_gdf.area
    polygons_gdf["partition"] = 1

    # create gbif_gdf, using polygon centroids
    species_list = [f"species_{i}" for i in range(nspecies)]

    # create a geopandas dataframe that maps each species to each centroid of polygons in polygons gdf
    df_list = []
    for poly in polygons_gdf.geometry:
        df = pd.DataFrame()
        df["speciesKey"] = species_list + species_list
        df["geometry"] = poly.centroid
        df_list.append(df)

    gbif_gdf = gpd.GeoDataFrame(pd.concat(df_list), crs=polygons_gdf.crs)

    savepath = Path(__file__).parent / Path(".test/")

    process_partition(gbif_gdf, polygons_gdf, savepath, batchsize)

    mygdf = gpd.read_parquet(savepath / Path("partition_1.parquet"))

    assert all(mygdf.sr == nspecies)


def test_lc_raster_to_multipoint():
    # Create a mock binary raster with rioxarray
    data = np.array([[0, 1], [1, 0]])  # A simple binary raster
    raster = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.array([0,1]), "x": np.array([0,1])}
    )

    result = lc_raster_to_multipoint(raster)
    expected_points = MultiPoint([(1, 0), (0, 1)])

    # Verify the result matches the expected points
    assert result.equals(
        expected_points
    ), "The MultiPoint object does not match the expected result."
