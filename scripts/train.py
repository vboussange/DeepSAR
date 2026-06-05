"""
Training MuScaRi models on CV folds.
"""
import json
import logging
import os
import socket
import torch
import torch.nn as nn
import pytorch_lightning as pl
import geopandas as gpd
from pathlib import Path

from muscari.muscari_ensemble import MuScaRiEnsemble
from muscari.muscari import MuScaRi
from muscari.trainer import MuScaRiLitModule, TrainConfig
from muscari.dataset import create_dataloader
from muscari.utils import (
    add_effort_columns,
    choose_accelerator,
    compute_metrics,
    evaluate_lit_model,
    feature_sets,
    finish_wandb,
    get_git_hash,
    log_wandb_metrics,
    make_trainer,
    maybe_wandb_logger,
    symmetric_arch,
)

SBCV_SAMPLES_PATH = Path(__file__).parent / "../data/processed/training_samples/sbcv/ceacce0"
BIOCLIMATE_VARS = [
    "bio1",
    "pet_penman_mean",
    "sfcWind_mean",
    "bio4",
    "rsds_1981-2010_range_V.2.1",
    "bio12",
    "bio15",
]

FEATURE_GROUP = "env_area"
FEATURE_CONFIGS = {
    "env_area": {
        "bioclimate_vars": [
            "bio1",
            "pet_penman_mean",
            "sfcWind_mean",
            "bio12",
        ],
        "include_elevation": True,
        "include_landcover": False,
    },
    "env_area_full_lc": {
        "bioclimate_vars": BIOCLIMATE_VARS,
        "include_elevation": True,
        "include_landcover": True,
    },
    "env_area_full_no_lc": {
        "bioclimate_vars": BIOCLIMATE_VARS,
        "include_elevation": True,
        "include_landcover": False,
    },
    "env_area_full_lc_no_dem": {
        "bioclimate_vars": BIOCLIMATE_VARS,
        "include_elevation": False,
        "include_landcover": True,
    },
}
SELECTED_FEATURE_CONFIG = "env_area"

EFFORT_TRANSFORM = "absolute"
ASYMPTOTE_TRANSFORM = "identity"
WEIBULL_PARAMETERIZATION = "legacy"
TARGET_TRANSFORM = "maxabs"
SELECTED_ARCHITECTURE_NAME = (
    f"{WEIBULL_PARAMETERIZATION}_{ASYMPTOTE_TRANSFORM}_{EFFORT_TRANSFORM}"
)
TRAIN_NAME = f"{SELECTED_FEATURE_CONFIG}_{SELECTED_ARCHITECTURE_NAME}"
RUN_ID = get_git_hash(short=True)
RUN_FOLDER = Path(__file__).parent / f"results/train/{TRAIN_NAME}/{RUN_ID}"

USE_WANDB = True
WANDB_PROJECT = "muscari-third-revision"
WANDB_GROUP = f"train_{RUN_ID}"
WANDB_TAGS = ["train", "third-revision", RUN_ID]

SMOKE_TEST = os.environ.get("MUSCARI_SMOKE_TEST", "0") == "1"
FOLD_IDS = [0] if SMOKE_TEST else list(range(5))
N_EPOCHS = 1 if SMOKE_TEST else 100
TRAIN_FRAC = 0.002 if SMOKE_TEST else 1.0

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: move to utils.py
def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def fold_files(config, fold_ids):
    return {
        str(fold_id): {
            split: str(config.path_sbcv_data / f"fold_{fold_id}_{split}.parquet")
            for split in ["train", "val", "test"]
        }
        for fold_id in fold_ids
    }


def resolve_feature_config() -> dict:
    try:
        return FEATURE_CONFIGS[SELECTED_FEATURE_CONFIG]
    except KeyError as exc:
        valid = ", ".join(sorted(FEATURE_CONFIGS))
        raise ValueError(
            f"Unknown feature config '{SELECTED_FEATURE_CONFIG}'. Expected one of: {valid}"
        ) from exc


def training_config_payload(config, feature_names, fold_summaries):
    metadata_files = [
        str(path)
        for path in [
            config.path_sbcv_data / "metadata.json",
            config.path_sbcv_data / "config_used.json",
        ]
        if path.exists()
    ]
    return json_ready({
        "run": {
            "script": str(Path(__file__)),
            "hostname": socket.gethostname(),
            "git_hash": get_git_hash(),
            "smoke_test": SMOKE_TEST,
            "run_folder": config.run_folder,
            "checkpoint_pattern": str(config.run_folder / "fold_<fold_id>.pth"),
            "ensemble_pretrained_path": str(config.run_folder / "ensemble_pretrained"),
        },
        "dataset": {
            "sbcv_dataset_id": config.path_sbcv_data.name,
            "sbcv_path": config.path_sbcv_data,
            "metadata_files": metadata_files,
            "fold_files": fold_files(config, FOLD_IDS),
        },
        "model": {
            "model_family": "MuScaRi",
            "architecture_variant": SELECTED_ARCHITECTURE_NAME,
            "layer_sizes": config.layer_sizes,
            "batchnorm": config.muscari_batchnorm,
            "asymptote_transform": ASYMPTOTE_TRANSFORM,
            "weibull_parameterization": WEIBULL_PARAMETERIZATION,
            "effort_transform": EFFORT_TRANSFORM,
            "target_transform": TARGET_TRANSFORM,
        },
        "training": {
            "train_config": vars(config),
            "seed": config.seed,
            "fold_ids": FOLD_IDS,
            "n_epochs": config.n_epochs,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "lr_scheduler_factor": config.lr_scheduler_factor,
            "lr_scheduler_patience": config.lr_scheduler_patience,
            "train_frac": TRAIN_FRAC,
            "loss": "MSELoss",
        },
        "features_and_labels": {
            "feature_group": FEATURE_GROUP,
            "selected_feature_config": SELECTED_FEATURE_CONFIG,
            "feature_config": resolve_feature_config(),
            "feature_columns": feature_names,
            "model_input_columns": ["log_observed_area"] + feature_names,
            "target_column": "sr",
            "derived_effort_column": "log_observed_area",
        },
        "wandb": {
            "enabled": USE_WANDB,
            "project": WANDB_PROJECT,
            "group": WANDB_GROUP,
            "tags": WANDB_TAGS,
        },
        "fold_summaries": fold_summaries,
    })


def write_training_config(config, feature_names, fold_summaries):
    config_path = config.run_folder / "config.json"
    with open(config_path, "w") as f:
        json.dump(training_config_payload(config, feature_names, fold_summaries), f, indent=2)
    logger.info(f"Wrote training config to {config_path}")


def make_wandb_logger(fold_id, feature_names, config):
    return maybe_wandb_logger(
        use_wandb=USE_WANDB,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        tags=WANDB_TAGS,
        name=f"{SBCV_SAMPLES_PATH.name}_fold_{fold_id}",
        config={
            "dataset_id": SBCV_SAMPLES_PATH.name,
            "fold": fold_id,
            "feature_names": feature_names,
            "architecture_variant": SELECTED_ARCHITECTURE_NAME,
            "effort_transform": EFFORT_TRANSFORM,
            "asymptote_transform": ASYMPTOTE_TRANSFORM,
            "weibull_parameterization": WEIBULL_PARAMETERIZATION,
            "target_transform": TARGET_TRANSFORM,
            "feature_set": SELECTED_FEATURE_CONFIG,
            "n_epochs": config.n_epochs,
            "batch_size": config.batch_size,
            "train_frac": TRAIN_FRAC,
            "run_folder": str(config.run_folder),
        },
    )


def save_ensemble_pretrained(run_folder: Path):
    ensemble_path = run_folder / "ensemble_pretrained"
    logger.info(f"Building ensemble from checkpoints in {run_folder}")
    ensemble = MuScaRiEnsemble.from_folds(run_folder, device="cpu")
    logger.info(f"Saving ensemble pretrained model to {ensemble_path}")
    ensemble.save_pretrained(ensemble_path)
    return ensemble_path


def train_fold(config: TrainConfig, fold_id, feature_names):
    logger.info(f"Training Fold {fold_id}...")
    pl.seed_everything(config.seed + fold_id)
    
    train_path = config.path_sbcv_data / f"fold_{fold_id}_train.parquet"
    val_path = config.path_sbcv_data / f"fold_{fold_id}_val.parquet"
    test_path = config.path_sbcv_data / f"fold_{fold_id}_test.parquet"
    
    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        logger.warning(f"Fold {fold_id} data not found. Skipping.")
        return

    train_df = gpd.read_parquet(train_path)
    val_df = gpd.read_parquet(val_path)
    test_df = gpd.read_parquet(test_path)
    raw_rows = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }
    if TRAIN_FRAC < 1.0:
        train_df = train_df.sample(frac=TRAIN_FRAC, random_state=config.seed + fold_id)
    sampled_rows = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }
    
    train_df = add_effort_columns(train_df, EFFORT_TRANSFORM)
    val_df = add_effort_columns(val_df, EFFORT_TRANSFORM)
    test_df = add_effort_columns(test_df, EFFORT_TRANSFORM)
    for df in [train_df, val_df, test_df]:
        df.dropna(subset=["log_observed_area"] + feature_names, inplace=True)
    filtered_rows = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }
        
    # Create dataloaders
    train_loader, feature_scaler, target_scaler = create_dataloader(
        train_df,
        feature_names,
        config.batch_size,
        config.num_workers,
        target_transform=TARGET_TRANSFORM,
    )
    val_loader, _, _ = create_dataloader(
        val_df,
        feature_names,
        config.batch_size,
        config.num_workers,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_transform=TARGET_TRANSFORM,
        shuffle=False,
    )
    
    # Initialize model
    model = MuScaRi(config.layer_sizes, 
                    feature_names=feature_names,
                    feature_scaler=feature_scaler,
                    target_scaler=target_scaler,
                    ffnn_batchnorm=config.muscari_batchnorm,
                    asymptote_transform=ASYMPTOTE_TRANSFORM,
                    weibull_parameterization=WEIBULL_PARAMETERIZATION)
    
    _, _, eval_device = choose_accelerator()

    lit_model = MuScaRiLitModule(model, config, nn.MSELoss())
    wandb_logger = make_wandb_logger(fold_id, feature_names, config)

    trainer = make_trainer(config.n_epochs, wandb_logger, config.lr_scheduler_patience)
    trainer.fit(lit_model, train_loader, val_loader)

    test_loader, _, _ = create_dataloader(
        test_df,
        feature_names,
        config.batch_size,
        config.num_workers,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_transform=TARGET_TRANSFORM,
        shuffle=False,
    )

    y_true_train, y_pred_train = evaluate_lit_model(lit_model, train_loader, eval_device)
    y_true_val, y_pred_val = evaluate_lit_model(lit_model, val_loader, eval_device)
    y_true_test, y_pred_test = evaluate_lit_model(lit_model, test_loader, eval_device)

    metrics = {
        "train": compute_metrics(y_true_train, y_pred_train),
        "val": compute_metrics(y_true_val, y_pred_val),
        "test": compute_metrics(y_true_test, y_pred_test),
    }
    if wandb_logger is not None:
        log_wandb_metrics(wandb_logger, {
            f"{split}_{metric}": value
            for split, split_metrics in metrics.items()
            for metric, value in split_metrics.items()
        })
        finish_wandb(wandb_logger)
    
    # Save model
    save_path = config.run_folder / f"fold_{fold_id}.pth"
    logger.info(f"Saving model to {save_path}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_names": feature_names,
        "config": config,
        "metrics": metrics,
    }, save_path)
    return {
        "fold": fold_id,
        "checkpoint_path": str(save_path),
        "split_paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "rows": {
            "raw": raw_rows,
            "after_train_frac": sampled_rows,
            "after_dropna": filtered_rows,
        },
        "metrics": metrics,
    }

if __name__ == "__main__":
    RUN_FOLDER.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(path_sbcv_data=SBCV_SAMPLES_PATH,
                         num_workers=4,
                         n_epochs=N_EPOCHS,
                         muscari_batchnorm=False,
                         muscari_asymptote_transform=ASYMPTOTE_TRANSFORM,
                         muscari_weibull_parameterization=WEIBULL_PARAMETERIZATION,
                         effort_transform=EFFORT_TRANSFORM,
                         layer_sizes=symmetric_arch(6, base=128, factor=4),
                         run_folder=RUN_FOLDER)
    
    sample_file = next(config.path_sbcv_data.glob("*_train.parquet"))
    df = gpd.read_parquet(sample_file)

    feature_config = resolve_feature_config()
    feature_names = feature_sets(
        df,
        feature_config["bioclimate_vars"],
        include_elevation=feature_config["include_elevation"],
        include_landcover=feature_config["include_landcover"],
    )[FEATURE_GROUP]
    logger.info(
        "Training with feature config %s (%d features): %s",
        SELECTED_FEATURE_CONFIG,
        len(feature_names),
        feature_names,
    )

    fold_summaries = []
    write_training_config(config, feature_names, fold_summaries)
    for fold_id in FOLD_IDS:
        fold_summary = train_fold(config, fold_id, feature_names)
        if fold_summary is not None:
            fold_summaries.append(fold_summary)
            write_training_config(config, feature_names, fold_summaries)

    if fold_summaries:
        save_ensemble_pretrained(config.run_folder)
