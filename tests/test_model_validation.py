import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.model_validation import get_spatial_block_cv_index

def test_get_spatial_block_cv_index():
    # Create a test DataFrame
    npartitions = 10
    df = pd.DataFrame({"partition": np.random.randint(0, npartitions, 100)})

    # Instantiate KFold
    k = 5
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    # Call the function under test
    cv_idxs_list = get_spatial_block_cv_index(df, kfold)

    # Assert that the number of folds is correct
    assert len(cv_idxs_list) == k, f"Expected {k} folds, got {len(cv_idxs_list)}"

    # Additional checks can include:
    # - Each fold's test indices do not appear in its training indices
    # - Each partition ID appears in exactly one test set across all folds
    partitions_in_test = []
    for train_idxs, test_idxs in cv_idxs_list:
        assert not set(train_idxs).intersection(set(test_idxs)), "Train and test indices overlap"
        test_partitions = df.loc[test_idxs, "partition"].unique()
        assert not set(test_partitions).intersection(partitions_in_test), "Partition ID appears in more than one test set"
        partitions_in_test.extend(test_partitions)