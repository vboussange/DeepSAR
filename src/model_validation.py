import pandas as pd
import numpy as np

npartitions = 100
df = pd.DataFrame({"partition" : np.random.randint(0, npartitions-1, 1000)})


def get_spatial_block_cv_index(gdf, kfold):
    """
    Splits a dataset into spatially-aware cross-validation folds.
    
    Parameters:
    - gdf (pandas.DataFrame): DataFrame with a "partition" column indicating spatial partition IDs.
    - kfold (KFold): Cross-validator defining the number of splits and shuffling strategy.
    
    Returns:
    - list of tuples: Each tuple contains two arrays for training and testing indices per fold.
    """
    partition_ids = gdf["partition"].unique()
    cv_idxs_list = []
    for train_idx, test_idx in kfold.split(partition_ids):
        test_partition_ids = partition_ids[test_idx]
        test_idxs = gdf.index[gdf.partition.isin(test_partition_ids)]
        train_idxs = gdf.index[~gdf.partition.isin(test_partition_ids)]
        cv_idxs_list.append((train_idxs, test_idxs))
    return cv_idxs_list
        
def get_habitat_spatial_block_cv_index(gdf, kfold, habitat):
    """
    Splits a dataset into spatially-aware cross-validation folds, filtered by habitat.
    
    Parameters:
    - gdf (pandas.DataFrame): DataFrame with "partition" and "habitat_id" columns.
    - kfold (KFold): Cross-validator for splits.
    - habitat (int/str): Habitat ID to filter test data.
    
    Returns:
    - list of tuples: Each tuple has training and testing indices, with tests filtered by habitat.
    """
    partition_ids = gdf["partition"].unique()
    cv_idxs_list = []
    for train_idx, test_idx in kfold.split(partition_ids):
        test_partition_ids = partition_ids[test_idx]
        test_idxs = gdf.index[np.logical_and(gdf.partition.isin(test_partition_ids), gdf.habitat_id == habitat)]
        train_idxs = gdf.index[~gdf.partition.isin(test_partition_ids)]
        cv_idxs_list.append((train_idxs, test_idxs))
    return cv_idxs_list
    