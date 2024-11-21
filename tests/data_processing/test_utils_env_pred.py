from src.data_processing.utils_env_pred import CHELSADataset

def test_CHELSADataset():
    dataset = CHELSADataset()
    env_pred = dataset.load()
    for var in env_pred["variable"].to_series():
        print(var)
        raster = env_pred.sel(variable=var)
        # checking if we have at least some data
        assert raster.isnull().sum().item() < raster.size