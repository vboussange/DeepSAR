from __future__ import annotations

import hashlib
import json
import logging
import socket
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

from muscari.dataset import create_dataloader
from muscari.utils import (
    add_effort_columns,
    compute_log1p_metrics,
    compute_metrics,
    evaluate_lit_model,
    finish_wandb,
    get_git_hash,
    json_ready,
    log_wandb_metrics,
    maybe_wandb_logger,
    residual_bias_slope,
    symmetric_arch,
)


logger = logging.getLogger(__name__)


def _discover_devices() -> list[str]:
    if torch.cuda.is_available():
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]


def _config_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _device_settings(device: str) -> tuple[str, int | list[int]]:
    if "cuda" in device:
        if ":" in device:
            return "gpu", [int(device.split(":", maxsplit=1)[1])]
        return "gpu", 1
    if "mps" in device:
        return "mps", 1
    return "cpu", 1


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


@dataclass
class TrainConfig:
    devices: list[str] = field(default_factory=list)
    seed: int = 1
    batch_size: int = 1024
    num_workers: int = 0
    n_epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10
    climate_variables: list[str] = field(default_factory=list)
    run_root: Path | None = None
    run_folder: Path | None = None
    path_sbcv_data: Path | None = None
    path_gift_data: Path | None = None
    muscari_batchnorm: bool = False
    muscari_asymptote_transform: str = "identity"
    muscari_weibull_parameterization: str = "legacy"
    effort_transform: str = "absolute"
    target_transform: str = "maxabs"
    layer_sizes: list[int] = field(
        default_factory=lambda: symmetric_arch(6, base=32, factor=4)
    )
    torch_num_threads: int | None = None
    fold_ids: list[int] = field(default_factory=lambda: list(range(5)))
    use_wandb: bool = False
    wandb_project: str = "muscari-third-revision"
    wandb_group: str = "train"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_config: dict[str, Any] = field(default_factory=dict)
    save_checkpoints: bool = False
    write_summary: bool = False
    use_validation_weights: bool = False
    overwrite: bool = False
    model_family: str = "MuScaRi"
    architecture_variant: str = "unspecified"
    feature_set: str | None = None
    feature_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    config_hash: str | None = None

    def __post_init__(self):
        if not self.devices:
            self.devices = _discover_devices()
        if self.path_sbcv_data is not None:
            self.path_sbcv_data = Path(self.path_sbcv_data)
        if self.path_gift_data is not None:
            self.path_gift_data = Path(self.path_gift_data)
        if self.run_root is not None:
            self.run_root = Path(self.run_root)
        if self.run_folder is not None:
            self.run_folder = Path(self.run_folder)


class MuScaRiLitModule(pl.LightningModule):
    def __init__(self, model, config, loss_fn):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        pl.seed_everything(config.seed)
        self.devices = config.devices
        self.sbcv_path = config.path_sbcv_data
        self.gift_path = config.path_gift_data
        self.gift_df = self._load_gift()
        self._configured_run_folder = config.run_folder

    def _load_gift(self):
        if self.gift_path is None:
            return None
        gift_df = gpd.read_parquet(self.gift_path)
        gift_df = self._add_effort_columns(gift_df)
        gift_df = gift_df.replace([np.inf, -np.inf], np.nan)
        # Keep a common complete-case cohort across benchmark feature sets.
        gift_df.dropna(inplace=True)
        return gift_df

    def _add_effort_columns(self, df):
        return add_effort_columns(df, self.config.effort_transform)

    def _default_model_metadata(self) -> dict[str, Any]:
        return {
            "model_family": self.config.model_family,
            "architecture_variant": self.config.architecture_variant,
            "layer_sizes": self.config.layer_sizes,
            "batchnorm": self.config.muscari_batchnorm,
            "asymptote_transform": self.config.muscari_asymptote_transform,
            "weibull_parameterization": self.config.muscari_weibull_parameterization,
            "effort_transform": self.config.effort_transform,
            "target_transform": self.config.target_transform,
        }

    def _canonical_payload(
        self,
        experiment_name: str,
        feature_names: list[str],
        train_frac: float,
        model_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "dataset": {
                "sbcv_dataset_id": self.sbcv_path.name if self.sbcv_path else None,
                "sbcv_path": self.sbcv_path,
                "gift_dataset_id": self.gift_path.parent.name if self.gift_path else None,
                "gift_path": self.gift_path,
                "gift_complete_case_policy": "all_columns" if self.gift_path else None,
            },
            "experiment": experiment_name,
            "features": {
                "feature_set": self.config.feature_set,
                "feature_config": self.config.feature_config,
                "feature_names": feature_names,
            },
            "model": model_metadata,
            "training": {
                "seed": self.config.seed,
                "batch_size": self.config.batch_size,
                "num_workers": self.config.num_workers,
                "n_epochs": self.config.n_epochs,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "lr_scheduler_factor": self.config.lr_scheduler_factor,
                "lr_scheduler_patience": self.config.lr_scheduler_patience,
                "fold_ids": self.config.fold_ids,
                "train_frac": train_frac,
                "effort_transform": self.config.effort_transform,
                "target_transform": self.config.target_transform,
                "use_validation_weights": self.config.use_validation_weights,
            },
            "metadata": self.config.metadata,
        }

    def _run_context(
        self,
        experiment_name: str,
        feature_names: list[str],
        train_frac: float,
        model_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        resolved_model_metadata = {
            **self._default_model_metadata(),
            **(model_metadata or {}),
        }
        canonical = self._canonical_payload(
            experiment_name,
            feature_names,
            train_frac,
            resolved_model_metadata,
        )
        config_hash = _config_hash(canonical)
        self.config.config_hash = config_hash
        return {
            "experiment_name": experiment_name,
            "feature_names": feature_names,
            "train_frac": train_frac,
            "model_metadata": resolved_model_metadata,
            "canonical": canonical,
            "config_hash": config_hash,
        }

    def _artifact_enabled(self) -> bool:
        return (
            self.config.save_checkpoints
            or self.config.write_summary
        )

    def _resolve_run_folder(self, config_hash: str) -> Path | None:
        if not self._artifact_enabled():
            return None
        if self._configured_run_folder is None:
            run_root = self.config.run_root or Path("scripts/results/train")
            self.config.run_folder = Path(run_root) / config_hash
        else:
            self.config.run_folder = self._configured_run_folder
        self.config.run_folder = Path(self.config.run_folder)
        self.config.run_folder.mkdir(parents=True, exist_ok=True)
        return self.config.run_folder

    def _check_existing_artifacts(self, run_folder: Path):
        if self.config.overwrite:
            return
        blockers = []
        if self.config.save_checkpoints:
            blockers.extend(run_folder.glob("fold_*.pth"))
        if self.config.write_summary and (run_folder / "config.json").exists():
            blockers.append(run_folder / "config.json")
        if blockers:
            listed = ", ".join(str(path) for path in blockers[:5])
            raise FileExistsError(
                f"Existing training artifacts found in {run_folder}: {listed}. "
                "Set overwrite=True or choose a different config/run folder."
            )

    def _fold_files(self) -> dict[str, dict[str, str]]:
        if self.sbcv_path is None:
            return {}
        return {
            str(fold_id): {
                split: str(self.sbcv_path / f"fold_{fold_id}_{split}.parquet")
                for split in ["train", "val", "test"]
            }
            for fold_id in self.config.fold_ids
        }

    def _metadata_files(self) -> list[str]:
        if self.sbcv_path is None:
            return []
        return [
            str(path)
            for path in [
                self.sbcv_path / "metadata.json",
                self.sbcv_path / "config_used.json",
            ]
            if path.exists()
        ]

    def _summary_payload(
        self,
        run_context: dict[str, Any],
        fold_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        run_folder = self.config.run_folder
        payload = {
            "run": {
                "hostname": socket.gethostname(),
                "git_hash": get_git_hash(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config_hash": run_context["config_hash"],
                "run_folder": run_folder,
                "checkpoint_pattern": (
                    run_folder / "fold_<fold_id>.pth" if run_folder else None
                ),
                "metadata": self.config.metadata,
            },
            "dataset": {
                "sbcv_dataset_id": self.sbcv_path.name if self.sbcv_path else None,
                "sbcv_path": self.sbcv_path,
                "gift_dataset_id": self.gift_path.parent.name if self.gift_path else None,
                "gift_path": self.gift_path,
                "gift_complete_case_policy": "all_columns" if self.gift_path else None,
                "metadata_files": self._metadata_files(),
                "fold_files": self._fold_files(),
            },
            "model": run_context["model_metadata"],
            "training": {
                "train_config": asdict(self.config),
                "seed": self.config.seed,
                "fold_ids": self.config.fold_ids,
                "n_epochs": self.config.n_epochs,
                "batch_size": self.config.batch_size,
                "num_workers": self.config.num_workers,
                "learning_rate": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "lr_scheduler_factor": self.config.lr_scheduler_factor,
                "lr_scheduler_patience": self.config.lr_scheduler_patience,
                "train_frac": run_context["train_frac"],
                "loss": "MSELoss",
                "target_transform": self.config.target_transform,
                "use_validation_weights": self.config.use_validation_weights,
            },
            "features_and_labels": {
                "feature_set": self.config.feature_set,
                "feature_config": self.config.feature_config,
                "feature_columns": run_context["feature_names"],
                "model_input_columns": ["log_observed_area"]
                + run_context["feature_names"],
                "target_column": "sr",
                "derived_effort_column": "log_observed_area",
            },
            "wandb": {
                "enabled": self.config.use_wandb,
                "project": self.config.wandb_project,
                "group": self.config.wandb_group,
                "tags": self.config.wandb_tags,
                "config": self.config.wandb_config,
            },
            "fold_summaries": fold_summaries,
        }
        return json_ready(payload)

    def _write_summary(
        self,
        run_context: dict[str, Any],
        fold_summaries: list[dict[str, Any]],
    ):
        if not self.config.write_summary or self.config.run_folder is None:
            return
        summary_path = self.config.run_folder / "config.json"
        with open(summary_path, "w") as handle:
            json.dump(
                self._summary_payload(run_context, fold_summaries),
                handle,
                indent=2,
            )
        logger.info("Wrote training summary to %s", summary_path)

    def _make_wandb_logger(
        self,
        experiment_name: str,
        fold_id: int,
        feature_names: list[str],
        train_frac: float,
        run_context: dict[str, Any],
    ):
        return maybe_wandb_logger(
            use_wandb=self.config.use_wandb,
            project=self.config.wandb_project,
            group=self.config.wandb_group,
            tags=self.config.wandb_tags + [experiment_name],
            name=f"{experiment_name}_{run_context['config_hash']}_fold_{fold_id}",
            config={
                "experiment": experiment_name,
                "fold": fold_id,
                "feature_names": feature_names,
                "train_frac": train_frac,
                "path_sbcv_data": str(self.config.path_sbcv_data),
                "path_gift_data": str(self.config.path_gift_data),
                "effort_transform": self.config.effort_transform,
                "target_transform": self.config.target_transform,
                "config_hash": run_context["config_hash"],
                "run_folder": str(self.config.run_folder),
                **self.config.wandb_config,
            },
        )

    @staticmethod
    def _wandb_run_id(wandb_logger) -> str | None:
        if wandb_logger is None:
            return None
        try:
            return str(wandb_logger.experiment.id)
        except Exception:
            return None

    def _train(self, model, train_loader, val_loader, loss_fn, device, wandb_logger=None):
        lit_model = MuScaRiLitModule(model, self.config, loss_fn)
        accelerator, devices = _device_settings(device)
        lightning_trainer = pl.Trainer(
            max_epochs=self.config.n_epochs,
            accelerator=accelerator,
            devices=devices,
            enable_checkpointing=False,
            logger=wandb_logger if wandb_logger is not None else False,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.lr_scheduler_patience * 2,
                )
            ],
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        lightning_trainer.fit(lit_model, train_loader, val_loader)
        # Retain stopping-epoch weights to reproduce the published training protocol.
        return lit_model

    def _read_fold_data(self, fold_id: int, train_frac: float):
        train_path = self.sbcv_path / f"fold_{fold_id}_train.parquet"
        val_path = self.sbcv_path / f"fold_{fold_id}_val.parquet"
        test_path = self.sbcv_path / f"fold_{fold_id}_test.parquet"
        split_paths = {"train": train_path, "val": val_path, "test": test_path}
        missing = [path for path in split_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Fold {fold_id} data not found at {self.sbcv_path}: {missing}"
            )

        train_df = gpd.read_parquet(train_path)
        val_df = gpd.read_parquet(val_path)
        test_df = gpd.read_parquet(test_path)
        raw_rows = {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        }
        if train_frac < 1.0:
            train_df = train_df.sample(
                frac=train_frac,
                random_state=self.config.seed + fold_id,
            )
        sampled_rows = {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        }
        return split_paths, raw_rows, sampled_rows, train_df, val_df, test_df

    def _prepare_df(self, df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        df = self._add_effort_columns(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["sr", "log_observed_area"] + feature_names)
        return df

    def _make_loaders(self, train_df, val_df, test_df, gift_df, feature_names, device):
        pin_memory = "cuda" in device
        train_loader, feature_scaler, target_scaler = create_dataloader(
            train_df,
            feature_names,
            self.config.batch_size,
            self.config.num_workers,
            target_transform=self.config.target_transform,
            pin_memory=pin_memory,
        )
        train_eval_loader, _, _ = create_dataloader(
            train_df,
            feature_names,
            self.config.batch_size,
            self.config.num_workers,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            target_transform=self.config.target_transform,
            shuffle=False,
            pin_memory=pin_memory,
        )
        val_loader, _, _ = create_dataloader(
            val_df,
            feature_names,
            self.config.batch_size,
            self.config.num_workers,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            target_transform=self.config.target_transform,
            shuffle=False,
            pin_memory=pin_memory,
        )
        test_loader, _, _ = create_dataloader(
            test_df,
            feature_names,
            self.config.batch_size,
            self.config.num_workers,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            target_transform=self.config.target_transform,
            shuffle=False,
            pin_memory=pin_memory,
        )
        gift_loader = None
        if gift_df is not None and not gift_df.empty:
            gift_loader, _, _ = create_dataloader(
                gift_df,
                feature_names,
                self.config.batch_size,
                self.config.num_workers,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                target_transform=self.config.target_transform,
                shuffle=False,
                pin_memory=pin_memory,
            )
        return (
            train_loader,
            train_eval_loader,
            val_loader,
            test_loader,
            gift_loader,
            feature_scaler,
            target_scaler,
        )

    def _evaluate_splits(self, lit_model, loaders, dfs, device):
        metrics_by_split = {}
        bias_slopes = {}
        flat_metrics = {}
        for split, loader in loaders.items():
            if loader is None:
                continue
            y_true, y_pred = evaluate_lit_model(lit_model, loader, device)
            split_metrics = {
                **compute_metrics(y_true, y_pred),
                **compute_log1p_metrics(y_true, y_pred),
            }
            metrics_by_split[split] = split_metrics
            bias_slopes[f"{split}_bias_slope_log_area"] = residual_bias_slope(
                y_true,
                y_pred,
                dfs[split]["log_sp_unit_area"].to_numpy(),
            )
            for key, value in split_metrics.items():
                flat_metrics[f"{split}_{key}"] = value
            flat_metrics[f"{split}_bias_slope_log_area"] = bias_slopes[
                f"{split}_bias_slope_log_area"
            ]

        if "test" in metrics_by_split:
            for key, value in metrics_by_split["test"].items():
                flat_metrics[f"interp_{key}"] = value
            flat_metrics["interp_bias_slope_log_area"] = bias_slopes[
                "test_bias_slope_log_area"
            ]
        if "gift" in metrics_by_split:
            for key, value in metrics_by_split["gift"].items():
                flat_metrics[f"extrap_{key}"] = value
            flat_metrics["extrap_bias_slope_log_area"] = bias_slopes[
                "gift_bias_slope_log_area"
            ]
        return metrics_by_split, bias_slopes, flat_metrics

    def _save_checkpoint(
        self,
        fold_id: int,
        lit_model: MuScaRiLitModule,
        feature_scaler,
        target_scaler,
        feature_names: list[str],
        metrics_by_split: dict[str, Any],
        flat_metrics: dict[str, Any],
        fold_summary: dict[str, Any],
        run_context: dict[str, Any],
    ) -> Path | None:
        if not self.config.save_checkpoints or self.config.run_folder is None:
            return None
        save_path = self.config.run_folder / f"fold_{fold_id}.pth"
        model = lit_model.model
        checkpoint = {
            "checkpoint_version": 2,
            "model_state_dict": _cpu_state_dict(model),
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_names": feature_names,
            "config": self.config,
            "metrics": metrics_by_split,
            "flat_metrics": flat_metrics,
            "fold_summary": fold_summary,
            "model_metadata": run_context["model_metadata"],
            "config_hash": run_context["config_hash"],
            "experiment": run_context["experiment_name"],
            "asymptote_transform": getattr(
                model,
                "asymptote_transform",
                self.config.muscari_asymptote_transform,
            ),
            "weibull_parameterization": getattr(
                model,
                "weibull_parameterization",
                self.config.muscari_weibull_parameterization,
            ),
        }
        torch.save(checkpoint, save_path)
        logger.info("Saved fold %s checkpoint to %s", fold_id, save_path)
        return save_path

    def _train_fold(
        self,
        fold_id: int,
        device: str,
        experiment_name: str,
        model_init,
        feature_names: list[str],
        train_frac: float,
        run_context: dict[str, Any],
    ):
        wandb_logger = None
        try:
            if self.config.torch_num_threads is not None:
                torch.set_num_threads(self.config.torch_num_threads)
            if "cuda" in device:
                torch.set_float32_matmul_precision("high")
            pl.seed_everything(self.config.seed + fold_id)

            split_paths, raw_rows, sampled_rows, train_df, val_df, test_df = self._read_fold_data(
                fold_id,
                train_frac,
            )
            train_df = self._prepare_df(train_df, feature_names)
            val_df = self._prepare_df(val_df, feature_names)
            test_df = self._prepare_df(test_df, feature_names)
            filtered_rows = {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            }

            gift_df = None
            if self.gift_df is not None:
                gift_df = self.gift_df.copy()
                gift_df = gift_df.dropna(subset=["sr", "log_observed_area"] + feature_names)

            loaders = self._make_loaders(
                train_df,
                val_df,
                test_df,
                gift_df,
                feature_names,
                device,
            )
            (
                train_loader,
                train_eval_loader,
                val_loader,
                test_loader,
                gift_loader,
                feature_scaler,
                target_scaler,
            ) = loaders

            model = model_init(
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
            )
            wandb_logger = self._make_wandb_logger(
                experiment_name,
                fold_id,
                feature_names,
                train_frac,
                run_context,
            )
            wandb_run_id = self._wandb_run_id(wandb_logger)
            lit_model = self._train(
                model,
                train_loader,
                val_loader,
                torch.nn.MSELoss(),
                device,
                wandb_logger,
            )
            eval_loaders = {
                "train": train_eval_loader,
                "val": val_loader,
                "test": test_loader,
                "gift": gift_loader,
            }
            eval_dfs = {
                "train": train_df,
                "val": val_df,
                "test": test_df,
                "gift": gift_df,
            }
            metrics_by_split, bias_slopes, flat_metrics = self._evaluate_splits(
                lit_model,
                eval_loaders,
                eval_dfs,
                device,
            )

            result_row = {
                "experiment": experiment_name,
                "fold": fold_id,
                "config_hash": run_context["config_hash"],
                "train_frac": train_frac,
                "n_train_samples": len(train_df),
                **flat_metrics,
            }
            fold_summary = {
                "fold": fold_id,
                "split_paths": json_ready(split_paths),
                "rows": {
                    "raw": raw_rows,
                    "after_train_frac": sampled_rows,
                    "after_dropna": filtered_rows,
                    "gift_after_dropna": len(gift_df) if gift_df is not None else None,
                },
                "metrics": metrics_by_split,
                "bias_slopes": bias_slopes,
                "flat_metrics": result_row,
                "wandb": {"run_id": wandb_run_id, "group": self.config.wandb_group},
            }
            checkpoint_path = self._save_checkpoint(
                fold_id,
                lit_model,
                feature_scaler,
                target_scaler,
                feature_names,
                metrics_by_split,
                result_row,
                fold_summary,
                run_context,
            )
            if checkpoint_path is not None:
                fold_summary["checkpoint_path"] = str(checkpoint_path)
            if wandb_logger is not None:
                log_wandb_metrics(wandb_logger, result_row)
            logger.info("Completed fold %s on device %s", fold_id, device)
            return {"row": result_row, "fold_summary": json_ready(fold_summary)}
        except Exception:
            logger.exception("Error processing fold %s on device %s", fold_id, device)
            raise
        finally:
            if wandb_logger is not None:
                try:
                    finish_wandb(wandb_logger)
                except Exception:
                    logger.exception("Failed to finish wandb run for fold %s", fold_id)

    def run(
        self,
        experiment_name: str,
        model_init,
        feature_names: list[str],
        train_frac: float = 1.0,
        model_metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        feature_names = list(feature_names)
        run_context = self._run_context(
            experiment_name,
            feature_names,
            train_frac,
            model_metadata,
        )
        run_folder = self._resolve_run_folder(run_context["config_hash"])
        if run_folder is not None:
            self._check_existing_artifacts(run_folder)

        fold_configs = [
            (fold_id, self.devices[index % len(self.devices)])
            for index, fold_id in enumerate(self.config.fold_ids)
        ]
        rows = []
        fold_summaries = []

        if len(self.devices) == 1 or len(fold_configs) == 1:
            logger.info("Running folds sequentially.")
            for fold_id, device in fold_configs:
                result = self._train_fold(
                    fold_id,
                    device,
                    experiment_name,
                    model_init,
                    feature_names,
                    train_frac,
                    run_context,
                )
                if result is not None:
                    rows.append(result["row"])
                    fold_summaries.append(result["fold_summary"])
                    self._write_summary(run_context, fold_summaries)
        else:
            logger.info(
                "Multiple devices detected (%s). Running folds in parallel.",
                len(self.devices),
            )
            max_workers = min(len(self.devices), len(fold_configs))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                future_to_fold = {
                    executor.submit(
                        self._train_fold,
                        fold_id,
                        device,
                        experiment_name,
                        model_init,
                        feature_names,
                        train_frac,
                        run_context,
                    ): fold_id
                    for fold_id, device in fold_configs
                }
                for future in as_completed(future_to_fold):
                    fold_id = future_to_fold[future]
                    try:
                        result = future.result()
                    except Exception:
                        logger.exception("Fold %s generated an exception", fold_id)
                        raise
                    if result is not None:
                        rows.append(result["row"])
                        fold_summaries.append(result["fold_summary"])
                        self._write_summary(run_context, fold_summaries)

        self._write_summary(run_context, fold_summaries)
        return pd.DataFrame(rows)
