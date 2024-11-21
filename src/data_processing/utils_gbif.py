"""
Defines functions to retrieve GBIF occurence data points
based on a specified bounding box, using pygbif
"""

import pygbif
from pygbif import occurrences, species
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.ops import transform
import pyproj
import pandas as pd

def get_plant_occurrences(long, lat, radius, **kwargs):
    nw_long, se_long, se_lat, nw_lat = calculate_bounding_box(long, lat, radius)

    # Query GBIF for plant occurrences
    results = occurrences.search(decimalLongitude=f'{nw_long},{se_long}',
                                 decimalLatitude=f'{se_lat},{nw_lat}',
                                 **kwargs) 
    return results

def get_unique_species(data):
    species_set = set()

    for occurrence in data['results']:
        # Assuming the species name is stored in 'species' key
        # Adjust the key as per the actual data structure
        species_name = occurrence.get('species')
        if species_name:
            species_set.add(species_name)
    return species_set
            
def plot_occurrences_within_bounding_box(long, lat, radius, occurrences_data):
    nw_long, se_long, se_lat, nw_lat = calculate_bounding_box(long, lat, radius)

    # Create a map with the PlateCarree projection
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot the bounding box
    box_lons = [nw_long, nw_long, se_long, se_long, nw_long]
    box_lats = [nw_lat, se_lat, se_lat, nw_lat, nw_lat]
    ax.plot(box_lons, box_lats, color='blue', linewidth=2, transform=ccrs.Geodetic())
    
    # Plot occurrences
    for occurrence in occurrences_data['results']:
        if 'decimalLongitude' in occurrence and 'decimalLatitude' in occurrence:
            plt.plot(occurrence['decimalLongitude'], occurrence['decimalLatitude'], 
                    'ro', markersize=2, transform=ccrs.Geodetic())

    # Set map extent
    ax.set_extent([nw_long - 1, se_long + 1, se_lat - 1, nw_lat + 1], crs=ccrs.PlateCarree())

    plt.title('Bounding Box Area')
    plt.show()


def get_species_list(intersection, crs_from, key):
    """
    Get species data within a given intersection polygon.

    Parameters:
    - intersection: The intersection polygon.
    - crs_from: The current CRS of the polygon.
    - crs_to: The target CRS for transformation.
    - key: Taxon key for species query.

    Returns:
    - A list of species within the given polygon.
    """
    project = pyproj.Transformer.from_crs(crs_from, 'EPSG:4326', always_xy=True).transform
    transformed_intersection = transform(project, intersection)
    data = occurrences.search(geometry=transformed_intersection.wkt, limit=300, taxonKey=key)
    return get_unique_species(data)


def create_species_area_dataframe(intersections, key, crs_from='EPSG:3035'):
    """
    Create a DataFrame with area, species richness, species list, and polygon details.

    Parameters:
    - intersections: List of intersection polygons.
    - key: Taxon key for species query.
    - crs_from: The current CRS of the polygons.
    - crs_to: The target CRS for transformation.

    Returns:
    - DataFrame with specified columns.
    """
    data_rows = []  # List to store each row as a dictionary
    for intersection in intersections:
        area = intersection.area
        species_list = get_species_list(intersection, crs_from, key)
        species_richness = len(species_list)
        data_rows.append({
            "area": area,
            "species_richness": species_richness,
            "species_list": species_list,
            "polygon_wkt": intersection.wkt
        })

    df = pd.DataFrame(data_rows)  # Convert the list of dictionaries to a DataFrame
    return df



if __name__ == "__main__":
    longitude = -0.1278 # Example longitude
    latitude = 51.5074 # Example latitude
    radius = 50 # Radius in kilometers

    key = species.name_backbone(name="Plantae")['usageKey']
    data = get_plant_occurrences(longitude, latitude, radius, taxonKey=key, limit=300, offset = 1000) 
    sp_list = get_unique_species(data)
    print(sp_list)
    plot_occurrences_within_bounding_box(longitude, latitude, radius, data)