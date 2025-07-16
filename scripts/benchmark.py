"""
This script benchmarks the `Neural4PWeibull` model on EVA and GIFT datasets
using various experimental configurations. It supports parallel execution across
multiple devices (CPU/GPU/MPS) and evaluates model performance using R2, D2,
RMSE, and MAPE. Results are saved as a CSV file for further analysis. 
"""

import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import multiprocessing as mp
from sklearn.metrics import (d2_absolute_error_score, root_mean_squared_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split

from deepsar.dataset import create_dataloader
from deepsar.neural_4pweibull import Neural4PWeibull
from deepsar.trainer import Trainer
from deepsar.utils import symmetric_arch
import warnings

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
class Config:
    devices: list
    seed: int = 1
    nruns: int = 5
    hash_data: str = HASH
    batch_size: int = 1024
    num_workers: int = 0
    n_epochs: int = 100
    val_size: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    climate_variables: list = field(
        default_factory=lambda: [
            "bio1",
            "pet_penman_mean",
            "sfcWind_mean",
            "bio4",
            "rsds_1981-2010_range_V.2.1",
            "bio12",
            "bio15",
        ]
    )
    run_name: str = "neural4p_weibull_nosmallmegaplots2_basearch6_" + HASH + "_benchmark"
    run_folder: Path = None
    path_eva_data: Path = None
    path_gift_data: Path = None

    def __post_init__(self):
        root = Path(__file__).parent
        self.run_folder = Path(root, 'results', f"{Path(__file__).stem}")
        self.run_folder.mkdir(parents=True, exist_ok=True)
        self.path_eva_data = (
            root
            / "../data"
            / "processed"
            / "EVA_CHELSA_compilation"
            / self.hash_data
            / "eva_chelsa_megaplot_data.parquet"
        )
        self.path_gift_data = (
            root
            / "../data"
            / "processed"
            / "GIFT_CHELSA_compilation"
            / "6c2d61d"
            / "megaplot_data.parquet"
        )


class Benchmark:
    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)

        # load EVA
        eva = pd.read_parquet(config.path_eva_data)
        for col in ("megaplot_area","observed_area"):
            eva[f"log_{col}"] = np.log(eva[col])
        eva = eva.replace([np.inf, -np.inf], np.nan).dropna()
        eva = eva[eva["num_plots"] > 2]  # TODO: to change
        self.eva = eva

        # load GIFT
        gift = gpd.read_parquet(config.path_gift_data)
        for col in ("megaplot_area","observed_area"):
            gift[f"log_{col}"] = np.log(gift[col])
        gift = gift.replace([np.inf, -np.inf], np.nan).dropna()
        self.gift = gift

        self.devices = config.devices
        self.nruns = config.nruns

    def run(self, predictors, loss_fn, layer_sizes, train_frac):
        # run in parallel on different GPUs
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=len(self.devices)) as pool:
            args = [(predictors, loss_fn, layer_sizes, train_frac, i) 
                   for i in range(self.nruns)]
            results = pool.starmap(self._single_run, args)
        
        # Count parameters once
        tmp = Neural4PWeibull(len(predictors) - 1, layer_sizes, torch.ones(4))
        num_params = sum(p.numel() for p in tmp.parameters())
        
        return {
            "logs": [r["log"] for r in results],
            "r2_eva": [r["r2_eva"] for r in results],
            "d2_eva": [r["d2_eva"] for r in results],
            "rmse_eva": [r["rmse_eva"] for r in results],
            "mape_eva": [r["mape_eva"] for r in results],
            "r2_gift": [r["r2_gift"] for r in results],
            "d2_gift": [r["d2_gift"] for r in results],
            "rmse_gift": [r["rmse_gift"] for r in results],
            "mape_gift": [r["mape_gift"] for r in results],
            "num_params": num_params,
        }        

    def _single_run(self, predictors, loss_fn, layer_sizes, train_frac, run_id):
        # assign a GPU
        device = self.devices[run_id % len(self.devices)]
        # split EVA into train/val/test
        eva_trainval = self.eva[self.eva["test"] == False].sample(
            frac=train_frac, random_state=run_id
        )
        eva_test = self.eva[self.eva["test"] == True]
        train_idx, val_idx = train_test_split(
            eva_trainval.index, test_size=self.config.val_size, random_state=run_id
        )
        df_tr = eva_trainval.loc[train_idx]
        df_val = eva_trainval.loc[val_idx]

        tr_loader, feat_s, targ_s = create_dataloader(
            df_tr,
            predictors,
            self.config.batch_size,
            self.config.num_workers,
        )
        val_loader, _, _ = create_dataloader(
            df_val,
            predictors,
            self.config.batch_size,
            self.config.num_workers,
            feat_s,
            targ_s,
        )

        # init model
        e0 = tr_loader.dataset.features[:, 0].median().item()
        c0 = tr_loader.dataset.targets.max().item()
        d0 = tr_loader.dataset.targets.min().item()
        p0 = [1e-1, c0, d0, e0]
        model = Neural4PWeibull(len(predictors) - 1, layer_sizes, p0).to(device)

        trainer = Trainer(
            config=self.config,
            model=model,
            feature_scaler=feat_s,
            target_scaler=targ_s,
            train_loader=tr_loader,
            val_loader=val_loader,
            compute_loss=loss_fn,
            device=device,
        )
        best_model, log = trainer.train(n_epochs=self.config.n_epochs)
        best_model.eval() # model is returned to CPU

        # EVA test
        X = torch.tensor(
            feat_s.transform(eva_test[predictors]), dtype=torch.float32
        ).to("cpu")
        with torch.no_grad():
            y_pred = (
                targ_s.inverse_transform(best_model(X).cpu().numpy())
                .squeeze()
            )
        y_true = eva_test["sr"].values
        r2_eva = r2_score(y_true, y_pred)
        d2_eva = d2_absolute_error_score(y_true, y_pred)
        rmse_eva = root_mean_squared_error(y_true, y_pred)
        mape_eva = mean_absolute_percentage_error(y_true, y_pred)

        # GIFT test (drop one feature if needed)
        Xg = torch.tensor(
            feat_s.transform(self.gift[predictors]), dtype=torch.float32
        ).to("cpu")
        Xg = Xg[:, 1:]  # e.g. drop log_megaplot_area if your predict_fn needs it
        with torch.no_grad():
            yg = best_model.predict_sr(Xg).cpu().numpy()
            y_pred_g = targ_s.inverse_transform(yg).squeeze()
        yg_true = self.gift["sr"].values
        r2_g = r2_score(yg_true, y_pred_g)
        d2_g = d2_absolute_error_score(yg_true, y_pred_g)
        rmse_g = root_mean_squared_error(yg_true, y_pred_g)
        mape_g = mean_absolute_percentage_error(yg_true, y_pred_g)

        return {
            "log": log,
            "r2_eva": r2_eva,
            "d2_eva": d2_eva,
            "rmse_eva": rmse_eva,
            "mape_eva": mape_eva,
            "r2_gift": r2_g,
            "d2_gift": d2_g,
            "rmse_gift": rmse_g,
            "mape_gift": mape_g,
        }


if __name__ == "__main__":
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():
        devices = ["mps"]
    else:
        devices = ["cpu"]
    config = Config(devices=devices)

    benchmark = Benchmark(config)

    # build experiments
    climate = config.climate_variables
    std_climate = ["std_" + v for v in climate]
    feats = climate + std_climate
    base_arch = symmetric_arch(6, base=32, factor=4)

    exps = []
    # area only
    exps.append(
        ("area", ["log_observed_area", "log_megaplot_area"], nn.MSELoss(), 1.0, base_arch)
    )
    # climate only
    exps.append(
        ("climate", ["log_observed_area"] + feats, nn.MSELoss(), 1.0, base_arch)
    )
    # area+climate with varying data fractions
    for frac in np.logspace(np.log10(1e-4), np.log10(1.0), 5):
        exps.append(
            ("area+climate", ["log_observed_area", "log_megaplot_area"] + feats, nn.MSELoss(), frac, base_arch)
        )
    # vary architecture
    for n in [0, 2, 4]:
        exps.append(
            ("area+climate",
             ["log_observed_area", "log_megaplot_area"] + feats,
             nn.MSELoss(),
             1.0,
             symmetric_arch(n, base=32, factor=4),
             )
        )

    rows = []
    for name, preds, loss_fn, frac, arch in exps:
        logger.info(f"Running {name}, frac={frac:.2f}, arch={arch}")
        out = benchmark.run(preds, loss_fn, arch, frac)
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