"""
Testing conversion from `epsg:4326` to `epsg:3035`, and testing units of distances and areas
"""

from shapely.geometry import Point
import geopandas as gpd
from src.data_processing.utils_polygons import place_randomly_rectangle
import numpy as np

# distance
test_points = gpd.GeoSeries([Point(17, 54), Point(18, 54)]) # Point(long, lat)
test_points.set_crs(epsg=4326, inplace=True)
test_points = test_points.to_crs(epsg=3035)
assert np.isclose(test_points.x.iat[1] - test_points.x.iat[0], 65.46e3,  atol = 1e3) #should be equal to 65.46*1e3 m (distance obtained from google maps)

# area
point = gpd.GeoSeries([Point(17, 54)], crs = "EPSG:4326").to_crs(epsg=3035).iat[0]
polygon = place_randomly_rectangle(point, 1000, 1000)

assert (gpd.GeoSeries([polygon], crs = "EPSG:3035").area == 1e6).all()