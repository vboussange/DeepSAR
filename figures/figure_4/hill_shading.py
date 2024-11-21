import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cbook, cm
from matplotlib.colors import LightSource
import rioxarray
import xarray as xr

from src.data_processing.utils_env_pred import CHELSADataset, CHELSA_PATH

climate_dataset = xr.open_dataset(CHELSADataset().cache_path)
climate_dataset = climate_dataset.rio.reproject("EPSG:3035")

# Load and format data
dem_path = '../../../data/DEM/dem_latlong.nc'
dem = rioxarray.open_rasterio(dem_path)


matched_dem = dem.rio.reproject_match(climate_dataset)
matched_dem = matched_dem.sel(band=1)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 9))
plt.setp(axs.flat, xticks=[], yticks=[])
ax = axs[0]
ls = LightSource(azdeg=315, altdeg=45)
cmap = plt.cm.gist_earth

ve = 0.1
ax.imshow(ls.hillshade(matched_dem.values, vert_exag=ve), cmap='gray')



# cookbook from https://matplotlib.org/stable/gallery/mplot3d/custom_shaded_3d_surface.html

# Load and format data
dem = cbook.get_sample_data('jacksboro_fault_dem.npz')
z = dem['elevation']
nrows, ncols = z.shape
x = np.linspace(dem['xmin'], dem['xmax'], ncols)
y = np.linspace(dem['ymin'], dem['ymax'], nrows)
x, y = np.meshgrid(x, y)

region = np.s_[5:50, 5:50]
x, y, z = x[region], y[region], z[region]

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

plt.show()