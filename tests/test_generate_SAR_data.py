import pandas as pd
import numpy as np
from pathlib import Path
from src.generate_sar_data_eva import (generate_SAR_data, 
                                   generate_SAR_data_SA,
                                   read_CLM5_data, 
                                   get_splot_bio_dfs, 
                                   apply_to_offdiag_elem,
                                   kfold_sar_data,
                                   format_clm5_for_training,
                                   average_min_pairwise_distance)
from shapely.geometry import MultiPoint
import time

from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def test_apply_to_offdiag_elem():
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    f = lambda x: sum(x)
    result = apply_to_offdiag_elem(f, A)
    assert result == 30
    
def test_read_CLM5_data():
    data_dir = Path("../../data/EVA/forest_t1")
    clm5_df = read_CLM5_data(data_dir)
    assert isinstance(clm5_df, pd.DataFrame)

def test_get_splot_bio_dfs():
    data_dir = Path("../../data/EVA/forest_t1")
    clm5_df, bio_df = get_splot_bio_dfs(data_dir)
    assert isinstance(clm5_df, pd.DataFrame)
    assert isinstance(bio_df, pd.DataFrame)

def test_generate_SAR_data_SA():
    data_dir = Path("../../data/EVA/forest_t1")
    clm5_df, bio_df = get_splot_bio_dfs(data_dir)
    data = generate_SAR_data_SA(clm5_df.iloc[:2], bio_df, npoints=1, max_aggregate=2, replace=False, stats_aggregate=["mean", "std", "heterogeneity", "distance"])

    # Test distance
    assert np.isclose(apply_to_offdiag_elem(np.mean,data["distance"][0]), 835, atol=1)
    
    assert np.isclose(data["a"][0], np.sum(clm5_df.iloc[:2]["a"]), atol=1)
    
def test_generate_SAR_data():
    data_dir = Path("../../data/EVA/forest_t1")
    clm5_df, bio_df = get_splot_bio_dfs(data_dir)
    data = generate_SAR_data(clm5_df.iloc[:2], bio_df, npoints=1, max_aggregate=2, replace=False, stats_aggregate=["mean", "std", "heterogeneity", "distance"])

    # Test distance
    assert np.isclose(data["distance_mean"][0], 835, atol=1)
    assert np.isclose(data["distance_std"][0], 0, atol=1)
    # Test area
    assert np.isclose(data["a"][0], np.sum(clm5_df["a"][:2]), atol=1)
    # Test mean
    assert np.isclose(data["RH2M_std_mean"][0], np.mean(clm5_df["RH2M_std"][0:2])) 
    
def test_kfold_sar_data():
    data_dir = Path("../../data/EVA/forest_t1")
    clm5_df, bio_df = get_splot_bio_dfs(data_dir)
    nsplits = 3
    cv = KFold(nsplits, shuffle = True)
    sub_clm5 = clm5_df.iloc[:30,:]
    data, idxs = kfold_sar_data(sub_clm5, 
                                bio_df, 
                                cv,
                                # npoints=1, 
                                max_aggregate=2, 
                                replace=False, 
                                stats_aggregate=["mean", "std", "heterogeneity", "distance"])
    
    len_train_test_fold = [len(train) + len(test) for train, test in cv.split(sub_clm5)]
    assert len(data) == sum(len_train_test_fold) * 2
    assert len(idxs) == nsplits
    assert idxs[-1][-1][-1] == len(data) - 1
    assert data.isna().any().any()
    
def test_format_clm5_for_training():
    data_dir = Path("../../data/EVA/forest_t1")
    clm5_df, _ = get_splot_bio_dfs(data_dir)
    data = format_clm5_for_training(clm5_df, stats_aggregate=["mean", "std", "distance"])
    
    assert np.isclose(data.a, clm5_df.a).all()
    assert np.isclose(data.sr, clm5_df.sr).all()
    _ns = [x for x in data.columns if "mean" in x]
    assert np.isclose(data[_ns[0:-1]], clm5_df.iloc[:,5:]).all()
    
def test_average_min_pairwise_distance():
    # simple test with random points
    num_points = 10000
    points = MultiPoint([(np.random.rand(), np.random.rand()) for _ in range(num_points)])
    average_distance = average_min_pairwise_distance(points)
    assert average_distance > 0.
    
    # testing spacing on a grid
    nx, ny = (100, 100)
    x = np.linspace(0, 99, nx)
    y = np.linspace(0, 99, ny)
    xv, yv = np.meshgrid(x, y)
    points = MultiPoint([(xy[0], xy[1]) for xy in zip(xv.flatten(), yv.flatten())])
    average_distance = average_min_pairwise_distance(points)
    assert np.isclose(average_distance, 1.)


