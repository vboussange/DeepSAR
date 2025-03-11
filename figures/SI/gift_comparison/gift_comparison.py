import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path
from rasterio.features import rasterize
import matplotlib.pyplot as plt

deepSAR_path = "/home/boussang/NNSAR/scripts/results/projections/SR_raster_2788m.tif"
deepSAR_raster = rioxarray.open_rasterio(deepSAR_path).squeeze("band")

cai_path = Path(__file__).parent / "../../../data/Cai2023/SR_Ensemble_rasterized.tif"
cai_raster = rioxarray.open_rasterio(cai_path, mask_and_scale=True).squeeze("band")
cai_raster = cai_raster.rio.reproject_match(deepSAR_raster)

fig, ax = plt.subplots()
cai_raster.plot(ax=ax, cbar_kwargs={'label': 'SR'})
ax.set_title("Cai et al. (2023), resolution: 88km")
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
fig.savefig("cai2023_88km.png")


fig, ax = plt.subplots()
deepSAR_raster.plot(ax=ax, cbar_kwargs={'label': 'SR'})
ax.set_title("Deep SAR model predictions, resolution: 2788m")
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
fig.savefig("deepsar_2788m.png")


deepSAR_path = "/home/boussang/NNSAR/scripts/results/projections/SR_raster_88170m.tif"
deepSAR_raster = rioxarray.open_rasterio(deepSAR_path).squeeze("band")


fig, ax = plt.subplots()
deepSAR_raster.plot(ax=ax, cbar_kwargs={'label': 'SR'})
ax.set_title("Deep SAR model predictions, resolution: 88km")
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
fig.savefig("deepsar_88km.png")
