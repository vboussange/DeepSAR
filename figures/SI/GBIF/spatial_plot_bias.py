"""
- Plotting spatial bias, i.e. cumulative count per number of partitions.
- Plotting same graph after spatial thinning
"""

# importing loading functions from compilation script
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent / Path("../../figure_2/")))
from figure_2_cinf import (
    process_results,
)

CLASSES = [
    " DBF_closed",
    " DBF_open",
    " ENF_closed",
    " ENF_open",
    " bare_sparse_vegetation",
    " herbaceous_vegetation",
    " herbaceous_wetland",
    " shrubland",
]

# loading
result_data = process_results()

result_data_gdf = result_data.gdf.groupby("habitat_names")


# spatial thinning
def sample_group(group, n):
    if len(group) > n:
        return group.sample(n, replace=False)
    else:
        return group
    
def do_thinning(df, n):
    return df.groupby("partition").apply(lambda group : sample_group(group, n))

fig, axs = plt.subplots(2, 4, figsize=(10, 6))
axs = axs.flatten()

for i, hab in enumerate(CLASSES):
    ax = axs[i]
    
    hab_gdf = result_data_gdf.get_group(hab)
    
    # do thinning
    
    partition_counts = (
        hab_gdf["partition"]
        .value_counts()
        .sort_values(ascending=False)
    )
    cumulative_counts = partition_counts.cumsum()
    
    n = partition_counts.quantile(0.75)
    thinned_hab_gdf = do_thinning(hab_gdf, int(n))
    thinned_partition_counts = (
        thinned_hab_gdf["partition"]
        .value_counts()
        .sort_values(ascending=False)
    )
    thinned_cumulative_counts = thinned_partition_counts.cumsum()

    print(f"Thinned df for {hab} has {len(thinned_hab_gdf)} rows")
    

    ax.plot(range(1, len(cumulative_counts) + 1), cumulative_counts.values, label="raw data")
    ax.plot(range(1, len(thinned_cumulative_counts) + 1), thinned_cumulative_counts.values, label="thinned data")

    ax.set_xlabel("Number of partitions")
    ax.set_ylabel("Cumulative number of entries")
    ax.set_title(hab)
    # ax.set_yscale("log")

fig.tight_layout()
fig.savefig("spatial_plot_bias.png", dpi=300, transparent=True)