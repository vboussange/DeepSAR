import random
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from sklearn.metrics import (d2_absolute_error_score, root_mean_squared_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split
from deepsar.dataset import create_dataloader
from deepsar.trainer import Trainer

import torch.multiprocessing as mp


@dataclass
class BenchmarkConfig:
    devices: list = field(default_factory=lambda: [])
    seed: int = 1
    nruns: int = 5
    hash_data: str = ""
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
    run_name: str = ""
    run_folder: Path = None
    path_eva_data: Path = None
    path_gift_data: Path = None

    def __post_init__(self):
        root = Path(__file__).parent
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


class Benchmarker:
    def __init__(self, config: BenchmarkConfig, gift: gpd.GeoDataFrame, eva: gpd.GeoDataFrame):
        self.config = config
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        self.eva = eva
        self.gift = gift

        self.devices = config.devices
        self.nruns = config.nruns

    def run(self, feature_names, loss_fn, model_init, train_frac):
        # Count parameters once
        tmp = model_init()
        num_params = sum(p.numel() for p in tmp.parameters())
        
        # run in parallel on different GPUs
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=len(self.devices)) as pool:
            args = [(feature_names, loss_fn, model_init, train_frac, i) 
                   for i in range(self.nruns)]
            results = pool.starmap(self._single_run, args)
        
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

    def _single_run(self, feature_names, loss_fn, model_init, train_frac, run_id):
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
            feature_names,
            self.config.batch_size,
            self.config.num_workers,
        )
        val_loader, _, _ = create_dataloader(
            df_val,
            feature_names,
            self.config.batch_size,
            self.config.num_workers,
            feat_s,
            targ_s,
        )

        # init model
        model = model_init(feature_names = feature_names,
                            feature_scaler=feat_s,
                            target_scaler=targ_s).to(device)

        trainer = Trainer(
            config=self.config,
            model=model,
            train_loader=tr_loader,
            val_loader=val_loader,
            compute_loss=loss_fn,
            device=device,
        )
        best_model, log = trainer.run(n_epochs=self.config.n_epochs)

        # EVA test
        y_pred = best_model.predict_sr(eva_test)
        y_true = eva_test["sr"].values
        r2_eva = r2_score(y_true, y_pred)
        d2_eva = d2_absolute_error_score(y_true, y_pred)
        rmse_eva = root_mean_squared_error(y_true, y_pred)
        mape_eva = mean_absolute_percentage_error(y_true, y_pred)

        # GIFT test (drop one feature if needed)
        y_pred_gift = best_model.predict_sr_tot(self.gift)
        yg_true = self.gift["sr"].values
        r2_g = r2_score(yg_true, y_pred_gift)
        d2_g = d2_absolute_error_score(yg_true, y_pred_gift)
        rmse_g = root_mean_squared_error(yg_true, y_pred_gift)
        mape_g = mean_absolute_percentage_error(yg_true, y_pred_gift)

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

