import geopandas as gpd
import numpy as np
from esda.moran import Moran
from libpysal.weights import DistanceBand
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from src.data_processing.utils_eva import EVADataset


def calculate_morans_i(gdf, distance_threshold):
    """
    Calculate Moran's I for the given GeoDataFrame and distance threshold.
    """
    w = DistanceBand.from_dataframe(gdf, threshold=distance_threshold, binary=True)
    moran = Moran(gdf['SR'].values, w)
    return moran.I, moran.p_sim

def determine_optimal_block_size(gdf, max_distance, num_steps=20):
    """
    Determine the optimal block size by calculating Moran's I for various distances.
    """
    distances = np.linspace(0, max_distance, num_steps)
    morans_i_values = []
    p_values = []

    for d in distances:
        if d > 0:  # Skip zero distance
            I, p = calculate_morans_i(gdf, d)
            morans_i_values.append(I)
            p_values.append(p)

    return distances[1:], morans_i_values, p_values

def plot_morans_i(distances, morans_i_values, p_values, alpha=0.05):
    """
    Plot Moran's I values against distances.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(distances, morans_i_values, label="Moran's I")
    plt.axhline(y=0, color='r', linestyle='--', label='No Autocorrelation')
    plt.xlabel('Distance (m)')
    plt.ylabel("Moran's I")
    plt.title("Moran's I vs Distance for Spatial Autocorrelation")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(distances, p_values, label="p-value")
    plt.axhline(y=alpha, color='r', linestyle='--', label=f'Alpha = {alpha}')
    plt.xlabel('Distance (m)')
    plt.ylabel("p-value")
    plt.title("p-value vs Distance for Moran's I")
    plt.legend()
    plt.show()

def main():
    gdf, dict_sp = EVADataset().load()
    gdf = gdf.sample(n=2000).to_crs("EPSG:3035")

    # Calculate pairwise distances
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    pairwise_dist = dist.pdist(coords, 'euclidean')
    max_distance = np.max(pairwise_dist) 

    # Determine optimal block size
    distances, morans_i_values, p_values = determine_optimal_block_size(gdf, max_distance)

    # Plot results
    plot_morans_i(distances, morans_i_values, p_values)

    # Determine the optimal distance where Moran's I is not significant (p > alpha)
    alpha = 0.05
    optimal_distance = next(dist for dist, p in zip(distances, p_values) if p > alpha)
    print(f"Optimal block size distance: {optimal_distance:.2f} m")

if __name__ == "__main__":
    main()
