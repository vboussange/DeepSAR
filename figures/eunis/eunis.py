import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon, Point, box
import numpy as np
from src.data_processing.utils_eunis import load_eunis_data, LEGEND_FILE_EUNIS, add_feature

gdf = load_eunis_data()
legend = pd.read_csv(LEGEND_FILE_EUNIS)
gdf = gdf[gdf.geometry.type=="Polygon"]

values = gdf.NAME.unique()
dict = {}
for _val in values:
    name = str(legend[legend.code -1 == int(_val)].EUNIS_l2.iloc[0])
    dict[_val] = name

gdf["habitat_l2"] = gdf['NAME'].replace(to_replace=dict)
gdf["habitat_l1"] = [s[0] for s in gdf['habitat_l2']]
print(gdf.habitat_l1.unique())

# Plot all forest habitats
crs_cpy = ccrs.epsg(3035)
fig, ax = plt.subplots(subplot_kw={'projection': crs_cpy})
add_feature(ax)
gdf.plot(ax=ax, column="habitat_l1", legend=True,)
ax.stock_img()
ax.set_extent(gdf.total_bounds)
fig.savefig("EUNIS_l1.png", dpi=300, transparent=True)

# Plot all forest habitats
crs_cpy = ccrs.epsg(3035)
fig, ax = plt.subplots(subplot_kw={'projection': crs_cpy})
add_feature(ax)
gdf.plot(ax=ax, column="habitat_l1", legend=True,)
# ax.stock_img()
# ax.set_extent(gdf.total_bounds)
fig.savefig("EUNIS_l1_nostock.png", dpi=300, transparent=True)