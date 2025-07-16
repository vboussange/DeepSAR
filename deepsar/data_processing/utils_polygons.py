import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import shape, box, Point, Polygon, MultiPolygon
from shapely.validation import make_valid
from geopy.distance import geodesic
from rasterio.features import shapes
from typing import Tuple
from tqdm import tqdm
from pathlib import Path
import xarray as xr
import rioxarray
from shapely.geometry import MultiPoint


def get_random_points_within_polygon(npoints, polygon):
    geominx, geominy, geomaxx, geomaxy = polygon.bounds
    points = gpd.GeoSeries()
    while len(points) < npoints:
        start_lon = np.random.uniform(geominx, geomaxx, npoints)
        start_lat = np.random.uniform(geominy, geomaxy, npoints)
        new_points = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(start_lon, start_lat)])
        pip = new_points.within(polygon)
        new_points_in = new_points[pip][0:min(npoints -
                                              len(points), pip.sum())]
        points = pd.concat([points, new_points_in])
    return points.reset_index(drop=True)


def get_random_points_within_polygon_gpu(npoints, polygon):
    try:
        import cuspatial
    except ImportError:
        raise ImportError("This function requires cuspatial for GPU acceleration. Please install cuspatial and try again.")
   
    geominx, geominy, geomaxx, geomaxy = polygon.bounds
    points = gpd.GeoSeries()
    while len(points) < npoints:
        start_lon = np.random.uniform(geominx, geomaxx, npoints)
        start_lat = np.random.uniform(geominy, geomaxy, npoints)
        new_points = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(start_lon, start_lat)])
        new_points_gpu = cuspatial.from_geopandas(new_points)
        polygon_gpu = cuspatial.from_geopandas(gpd.GeoSeries([polygon]))
        pip = cuspatial.point_in_polygon(new_points_gpu, polygon_gpu)
        new_points_in = new_points[
            pip[0].to_pandas()][0:min(npoints - len(points), pip[0].sum())]
        points = pd.concat([points, new_points_in])
    return points.reset_index()


def get_random_points_within_land(npoints, land_support_raster):
    """
    Similar to `get_random_points_within_polygon_gpu`, but working with rasters.
    Performs much faster, because the approach selects at random points that are already within land.
    """
    points = []
    # creating meshgrid
    y, x = xr.broadcast(land_support_raster.y, land_support_raster.x)
    # masking
    x, y = x.to_numpy(), y.to_numpy()

    x = x[land_support_raster.to_numpy() > 0]
    y = y[land_support_raster.to_numpy() > 0]

    for i in np.random.choice(range(len(x)), npoints):
        points.append(Point(x[i], y[i]))

    return gpd.GeoSeries(points)


def place_randomly_rectangle(point, length, height):
    start_lon = point.x
    start_lat = point.y

    rectangle_coords = [(start_lon, start_lat),
                        (start_lon + length, start_lat),
                        (start_lon + length, start_lat + height),
                        (start_lon, start_lat + height),
                        (start_lon, start_lat)]

    return Polygon(rectangle_coords)


def get_poly_range(gdf_polygon):
    gdf_polygon["area"] = gdf_polygon.area

    gdf_polygon = gdf_polygon.sort_values(by="area")
    minx, miny, maxx, maxy = gdf_polygon.geometry.iloc[0].bounds

    polyminx = maxx - minx
    polyminy = maxy - miny

    minx, miny, maxx, maxy = gdf_polygon.geometry.iloc[-1].bounds

    polymaxx = maxx - minx
    polymaxy = maxy - miny

    return (polyminx, polyminy, polymaxx, polymaxy)


def get_largest_component(polygon):
    if type(polygon) is MultiPolygon:
        return max(
            polygon.geoms, key=lambda a: a.area
        )  # https://gis.stackexchange.com/questions/318750/how-to-extract-biggest-polygon-from-multipolygon-in-geopandas
    else:
        return polygon


def size_from_template(poly, _):
    minx, miny, maxx, maxy = poly.bounds
    return maxx - minx, maxy - miny


def random_size_from_range(_, poly_range):
    polyminx, polyminy, polymaxx, polymaxy = poly_range
    length = np.random.uniform(polyminx, polymaxx)
    height = np.random.uniform(polyminy, polymaxy)
    return length, height


def generate_random_polygons(num_polygons,
                             dataset,
                             size_determination_func,
                             connectivity=4):
    """
    Creates a GeoDataFrame of random polygons based on polygon statistics derived from `dataset` based on `connectivity`.

    Parameters:
    num_polygons: The number of random rectangles to generate.
    dataset: the landcover dataset 

    Returns:
    GeoDataFrame with the specified number of random rectangles.
    """
    gdf_template = dataset.load_shapefile(connectivity)
    geo_bounds = gdf_template.total_bounds

    print("Generating polygon locations")
    land_support_raster = dataset.get_land_support(geo_bounds)

    # /!\ because we do not account for polygon length and height when generating the polygon location, it could be that the polygon will be cropped when located at the border of the zone considered
    points = get_random_points_within_land(num_polygons, land_support_raster)

    print("Generating polygons")
    polygons = []
    poly_range = get_poly_range(gdf_template)  # used by random_size_from_range
    for i, poly in tqdm(
            enumerate(np.random.choice(gdf_template.geometry, num_polygons))):
        length, height = size_determination_func(poly, poly_range)
        new_poly = place_randomly_rectangle(points[i], length, height)
        minx, miny, maxx, maxy = new_poly.bounds
        new_poly_raster = land_support_raster.sel(x=slice(minx, maxx),
                                                  y=slice(maxy, miny))

        max_area = -1
        new_poly = None
        # retaining largest Polygon, substracting non-land area
        for (s, v) in shapes(new_poly_raster.to_numpy(),
                             mask=None,
                             transform=new_poly_raster.rio.transform(),
                             connectivity=4):
            p = shape(s)
            if dataset.is_land(v) and p.area > max_area:
                new_poly = p
                max_area = p.area
        if new_poly:
            polygons.append(new_poly)

    gdf_habitat_free = gpd.GeoDataFrame({'geometry': polygons})
    gdf_habitat_free.crs = gdf_template.crs

    print("Cleaning up polygons...")
    gdf_habitat_free = gdf_habitat_free.loc[~gdf_habitat_free.geometry.
                                            is_empty]
    gdf_habitat_free.reset_index(drop=True, inplace=True)
    gdf_habitat_free.geometry = gdf_habitat_free.geometry.apply(make_valid)

    return gdf_habitat_free


def calculate_bounding_box(long, lat, radius):
    # Calculate bounding box coordinates
    northwest = geodesic(kilometers=radius).destination((lat, long), 315)
    southeast = geodesic(kilometers=radius).destination((lat, long), 135)
    return northwest[1], southeast[1], southeast[0], northwest[0]


def create_grid(gdf, block_length):
    """
    Creates a grid for spatial analysis.
    
    Args:
    gdf (GeoDataFrame): The geopandas dataframe to be used.
    block_length: the side of a block 

    Returns:
    GeoDataFrame: A geopandas dataframe representing the grid.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    x_edges = np.arange(minx-1e-5, maxx + 1e-5 + block_length, block_length)
    y_edges = np.arange(miny-1e-5, maxy + 1e-5 + block_length, block_length)

    grid_cells = []
    for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
        for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
            grid_cells.append(box(x0, y0, x1, y1))

    return gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=gdf.crs)


def partition_polygon_gdf(gdf, block_length):
    """
    Spatially split the data into independent partitions, for e.g. SBCV.
    """

    grid_gdf = create_grid(gdf, block_length)
    joined = gpd.sjoin(gdf, grid_gdf, how='left', predicate="intersects")
    gdf["partition"] = joined.index_right.astype(int)

    return gdf


def lc_raster_to_multipoint(binary_raster):
    """
    Converts a rioxarray object with binary values into a shapely MultiPoint
    object, where each point represents the geographic location of cells with a
    value of 1, within the raster's CRS.
    """
    
    y_indices, x_indices = np.where(binary_raster)
    points = [(binary_raster.x.values[x], binary_raster.y.values[y])
              for y, x in zip(y_indices, x_indices)]
    return MultiPoint(points)
