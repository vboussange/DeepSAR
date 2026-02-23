Original raster comes from https://www.eea.europa.eu/data-and-maps/data/eu-dem#tab-data-download, and is available at 25m resolution in EPSG:3035.
To resample the EEA DEM EPSG:3035 at 1000m resolution, use the following commands:
```
export GDAL_CACHEMAX=50%
export GDAL_NUM_THREADS=ALL_CPUS
gdalwarp -t_srs EPSG:3035 -tr 1000 1000 -r bilinear \
  -co COMPRESS=DEFLATE \
  -co PREDICTOR=3 \
  -co TILED=YES \
  eudem_dem_3035_europe.tif eudem_dem_3035_europe_1000m.tif
```
<!-- explain compression options -->

Please `eudem_dem_3035_europe_1000m.tif` in this folder.