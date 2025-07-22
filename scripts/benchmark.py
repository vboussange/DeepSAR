"""
This script benchmarks the `Deep4PWeibull` model on EVA and GIFT datasets
using various experimental configurations. It supports parallel execution across
multiple devices (CPU/GPU/MPS) and evaluates model performance using R2, D2,
RMSE, and MAPE. Results are saved as a CSV file for further analysis. 
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
from deepsar.utils import symmetric_arch
from deepsar.benchmarker import BenchmarkConfig, Benchmarker
from deepsar.deep4pweibull import Deep4PWeibull
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

HASH = "0b85791"

def setup_logger():
    log = logging.getLogger("benchmark")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        h.setFormatter(fmt)
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log

logger = setup_logger()

@dataclass
class Deep4PWeibullInit():
    feature_names: list
    architecture: list = field(default_factory=lambda: symmetric_arch(6, base=32, factor=4))
    def __call__(self, **kwargs):
        return Deep4PWeibull(feature_names=self.feature_names, 
                             layer_sizes=self.architecture, 
                             **kwargs)

if __name__ == "__main__":
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():
        devices = ["mps"]
    else:
        devices = ["cpu"]
    root_folder = Path(__file__).parent / Path('results', 'benchmark')
    root_folder.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig(devices=devices,
                             hash_data=HASH,
                            #  n_epochs=1, # for quick testing
                             run_folder = root_folder,
                             run_name="deep4pweibull_basearch6_" + HASH + "_benchmark")
    
    # load EVA
    eva = gpd.read_parquet(config.path_eva_data)
    eva = eva[eva["num_plots"] > 2]  # TODO: to change

    # load GIFT
    gift = gpd.read_parquet(config.path_gift_data)

    # preprocess data
    for data in (eva, gift):
        # replace inf and -inf with NaN, then drop NaN rows
        data["log_sp_unit_area"] = np.log(data["megaplot_area"])  # TODO: legacy name, to be changed in the future
        data["log_observed_area"] = np.log(data["observed_area"])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

    benchmark = Benchmarker(config, gift=gift, eva=eva)

    # build experiments
    climate = config.climate_variables
    std_climate = ["std_" + v for v in climate]
    climate_feats = climate + std_climate

    exps = []
    # area only
    exps.append(
        ("area", 
         nn.MSELoss(), 
         1.0, 
         Deep4PWeibullInit(feature_names=["log_sp_unit_area"]))
    )
    # climate only
    exps.append(
        ("climate", 
         nn.MSELoss(), 
         1.0, 
         Deep4PWeibullInit(feature_names=climate_feats))
    )
    # area+climate with varying data fractions
    for frac in np.logspace(np.log10(1e-4), np.log10(1.0), 5):
        exps.append(
            ("area+climate",
             nn.MSELoss(), 
             frac, 
             Deep4PWeibullInit(feature_names=["log_sp_unit_area"] + climate_feats))
        )
    # vary architecture
    for n in [0, 2, 4]:
        exps.append(
            ("area+climate",
             nn.MSELoss(),
             1.0,
             Deep4PWeibullInit(feature_names=["log_sp_unit_area"] + climate_feats,
                               architecture=symmetric_arch(n, base=32, factor=4))
             )
        )

    rows = []
    for name, loss_fn, frac, init in exps:
        logger.info(f"Running {name}, frac={frac:.2f}, model={init}")
        out = benchmark.run(loss_fn, init, frac)
        for run_id in range(benchmark.nruns):
            rows.append(
                {
                    "model": name,
                    "run": run_id,
                    "train_frac": frac,
                    "num_params": out["num_params"],
                    "r2_eva": out["r2_eva"][run_id],
                    "d2_eva": out["d2_eva"][run_id],
                    "rmse_eva": out["rmse_eva"][run_id],
                    "mape_eva": out["mape_eva"][run_id],
                    "r2_gift": out["r2_gift"][run_id],
                    "d2_gift": out["d2_gift"][run_id],
                    "rmse_gift": out["rmse_gift"][run_id],
                    "mape_gift": out["mape_gift"][run_id],
                }
            )
        logger.info(f"Finished {name}")

    df = pd.DataFrame(rows)
    out_csv = config.run_folder / f"{config.run_name}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved to {out_csv}")