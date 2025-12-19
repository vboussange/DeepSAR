import rioxarray
import xarray as xr

l

with rioxarray.open_rasterio(tiff_path, mask_and_scale=True) as da:
    # we load slightly more than the extent to be able to correctly interpolate
    dx = 0.1
    cda = da.rio.clip_box(minx=extent[0] - dx, miny=extent[1] - dx, maxx=extent[2]+ dx, maxy=extent[3]+ dx)
    cda = cda.sel(band=1)
    cda = cda.drop_vars(["band"]) # we keep `spatial_ref` var. as it contains crs data
    
    # extracting name and renaming
    name = tiff_path.stem.split("CHELSA_")[1].split("_1981-2010_V.2.1")[0]
    cda = cda.rename(name)
