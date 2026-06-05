import logging
import pickle
import socket

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import git

class MSELogLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELogLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, input, target):
        log_input = torch.log(torch.clamp(input, min=1e-8))
        log_target = torch.log(torch.clamp(target, min=1e-8))
        loss = (log_input - log_target) ** 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
        
def save_to_pickle(filepath, **kwargs):
    objects_dict = kwargs
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as file:
        pickle.dump(objects_dict, file)
    logging.info(f"Results saved at {filepath}")

def symmetric_arch(n, base=32, factor=2):
    half = (n + 1) // 2
    front = [base * factor**i for i in range(half)]
    mirror = front[:-1] if n % 2 else front
    return front + mirror[::-1]


def get_git_hash(short=True, fallback="unknown"):
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.git.rev_parse(repo.head, short=short)
    except git.InvalidGitRepositoryError:
        logging.warning("Could not determine git hash; using '%s'.", fallback)
        return fallback


def add_effort_columns(df: pd.DataFrame, effort_transform: str) -> pd.DataFrame:
    df = df.copy()
    df["log_sp_unit_area"] = np.log(df["sp_unit_area"])
    log_observed_area = np.log(df["observed_area"])
    if effort_transform == "absolute":
        df["log_observed_area"] = log_observed_area
    elif effort_transform == "relative":
        df["log_observed_area"] = log_observed_area / df["log_sp_unit_area"]
    else:
        raise ValueError("effort_transform must be 'absolute' or 'relative'")
    return df


def climate_dem_features(df: pd.DataFrame, bioclimate_vars: list[str]) -> list[str]:
    climate_feats = list(bioclimate_vars) + [f"std_{v}" for v in bioclimate_vars]
    dem_feats = ["elevation", "std_elevation"]
    return [c for c in climate_feats + dem_feats if c in df.columns]


def feature_sets(df: pd.DataFrame, bioclimate_vars: list[str]) -> dict[str, list[str]]:
    env = climate_dem_features(df, bioclimate_vars)
    return {
        "area": ["log_sp_unit_area"],
        "env": env,
        "env_area": env + ["log_sp_unit_area"],
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        d2_absolute_error_score,
        mean_absolute_percentage_error,
        r2_score,
        root_mean_squared_error,
    )

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    relative_bias = (y_pred - y_true) / y_true
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "d2": float(d2_absolute_error_score(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "mean_relative_bias": float(np.mean(relative_bias)),
        "median_relative_bias": float(np.median(relative_bias)),
    }


def residual_bias_slope(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_sp_unit_area: np.ndarray,
) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    relative_bias = (y_pred - y_true) / y_true
    finite = np.isfinite(relative_bias) & np.isfinite(log_sp_unit_area)
    if finite.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.asarray(log_sp_unit_area)[finite], relative_bias[finite], 1)[0])


def choose_accelerator() -> tuple[str, int | list[int], str]:
    if torch.cuda.is_available():
        return "gpu", 1, "cuda:0"
    if torch.backends.mps.is_available():
        return "mps", 1, "mps"
    return "cpu", 1, "cpu"


def evaluate_lit_model(lit_model, data_loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    lit_model.eval()
    lit_model.to(device)
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            y_pred = lit_model(x)
            preds.append(y_pred.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    if lit_model.model.target_scaler:
        preds = lit_model.model.target_scaler.inverse_transform(preds)
        targets = lit_model.model.target_scaler.inverse_transform(targets.reshape(-1, 1))

    return targets.flatten(), preds.flatten()


def maybe_wandb_logger(
    *,
    use_wandb: bool,
    project: str,
    group: str,
    tags: list[str],
    name: str,
    config: dict,
):
    if not use_wandb:
        return None
    from pytorch_lightning.loggers import WandbLogger

    return WandbLogger(
        project=project,
        group=group,
        tags=tags,
        name=name,
        config={
            **config,
            "git_hash": get_git_hash(),
            "hostname": socket.gethostname(),
        },
    )


def log_wandb_metrics(wandb_logger, metrics: dict[str, float]) -> None:
    if wandb_logger is None:
        return
    wandb_logger.experiment.log(metrics)


def finish_wandb(wandb_logger) -> None:
    if wandb_logger is None:
        return
    import wandb

    wandb.finish()


def make_trainer(max_epochs: int, wandb_logger, lr_scheduler_patience: int):
    import pytorch_lightning as pl

    accelerator, devices, _ = choose_accelerator()
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=False,
        logger=wandb_logger if wandb_logger is not None else False,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=lr_scheduler_patience * 2,
            )
        ],
        enable_progress_bar=False,
    )


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
