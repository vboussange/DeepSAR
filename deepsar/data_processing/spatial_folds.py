import geopandas as gpd
import numpy as np


def assign_checkerboard_folds(
    gdf: gpd.GeoDataFrame,
    n_splits: int = 5,
    block_size: int = 10_000,
) -> gpd.GeoDataFrame:
    """
    Assign spatial folds to a GeoDataFrame using a checkerboard pattern.

    Args:
        gdf: GeoDataFrame with point geometries.
        n_splits: Number of folds.
        block_size: Size of the checkerboard blocks in meters (projected CRS).

    Returns:
        GeoDataFrame with `grid_x`, `grid_y`, and `spatial_split` columns.
    """
    minx, miny, maxx, maxy = gdf.total_bounds

    grid_x = np.floor((gdf.geometry.x - minx) / block_size).astype(int)
    grid_y = np.floor((gdf.geometry.y - miny) / block_size).astype(int)

    gdf = gdf.copy()
    gdf["grid_x"] = grid_x
    gdf["grid_y"] = grid_y
    gdf["spatial_split"] = (gdf["grid_x"] + gdf["grid_y"]) % n_splits

    return gdf
