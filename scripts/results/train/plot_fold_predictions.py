"""
Plot predictions for each CV fold on train/val/test and GIFT datasets.

Debug script: adjust the constants below if needed, then run the file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import geopandas as gpd
import torch
import matplotlib.pyplot as plt

from muscari.dataset import create_dataloader
from muscari.muscari import MuScaRi

# Debug configuration (edit if needed)
RUN_DIR = Path("6dcd90c")
GIFT_PATH = Path(__file__).parent / "../../../data/processed/test_samples_GIFT/6dcd90c/compiled_data.parquet"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def _load_checkpoint(path: Path, device: str) -> dict:
    print(f"Loading checkpoint: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _prepare_df(df: gpd.GeoDataFrame, feature_names: list[str]) -> gpd.GeoDataFrame:
    df = df.copy()
    df["log_observed_area"] = np.log(df["observed_area"])
    df["log_sp_unit_area"] = np.log(df["sp_unit_area"])
    required = ["log_observed_area", "sr"] + feature_names
    df.dropna(subset=required, inplace=True)
    return df


def _build_loader(
    df: gpd.GeoDataFrame,
    feature_names: list[str],
    batch_size: int,
    num_workers: int,
    feature_scaler,
    target_scaler,
) -> torch.utils.data.DataLoader:
    loader, _, _ = create_dataloader(
        df,
        feature_names,
        batch_size,
        num_workers,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        shuffle=False,
    )
    return loader


def _predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    target_scaler,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)
            preds.append(y_pred.cpu())
            targets.append(yb.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    if target_scaler is not None:
        preds = target_scaler.inverse_transform(preds)
        targets = target_scaler.inverse_transform(targets)

    return targets.flatten(), preds.flatten()


def _plot_scatter(ax, y_true, y_pred, title: str):
    ax.scatter(y_true, y_pred, s=8, alpha=0.35)
    if y_true.size and y_pred.size:
        vmin = min(y_true.min(), y_pred.min())
        vmax = max(y_true.max(), y_pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], color="black", lw=1, alpha=0.7)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
    ax.set_title(title)
    ax.set_xlabel("Observed SR")
    ax.set_ylabel("Predicted SR")
    ax.grid(alpha=0.2)



if __name__ == "__main__":

    checkpoint_paths = sorted(RUN_DIR.glob("fold_*.pth"))
    out_dir = RUN_DIR / "debug_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path in checkpoint_paths:
        fold_id = ckpt_path.stem.split("_")[-1]
        checkpoint = _load_checkpoint(ckpt_path, DEVICE)
        feature_names = checkpoint["feature_names"]
        feature_scaler = checkpoint["feature_scaler"]
        target_scaler = checkpoint["target_scaler"]
        config = checkpoint["config"]

        sbcv_path = config.sbcv_path

        train_df = gpd.read_parquet(Path(sbcv_path) / f"fold_{fold_id}_train.parquet")
        val_df = gpd.read_parquet(Path(sbcv_path) / f"fold_{fold_id}_val.parquet")
        test_df = gpd.read_parquet(Path(sbcv_path) / f"fold_{fold_id}_test.parquet")
        gift_df = gpd.read_parquet(GIFT_PATH)

        train_df = _prepare_df(train_df, feature_names)
        val_df = _prepare_df(val_df, feature_names)
        test_df = _prepare_df(test_df, feature_names)
        gift_df = _prepare_df(gift_df, feature_names)
        gift_df["log_observed_area"] = 1e2

        train_loader = _build_loader(
            train_df,
            feature_names,
            config.batch_size,
            config.num_workers,
            feature_scaler,
            target_scaler,
        )
        val_loader = _build_loader(
            val_df,
            feature_names,
            config.batch_size,
            config.num_workers,
            feature_scaler,
            target_scaler,
        )
        test_loader = _build_loader(
            test_df,
            feature_names,
            config.batch_size,
            config.num_workers,
            feature_scaler,
            target_scaler,
        )
        gift_loader = _build_loader(
            gift_df,
            feature_names,
            config.batch_size,
            config.num_workers,
            feature_scaler,
            target_scaler,
        )

        model = MuScaRi.initialize(checkpoint, device=DEVICE)

        y_train, yhat_train = _predict(model, train_loader, target_scaler, DEVICE)
        y_val, yhat_val = _predict(model, val_loader, target_scaler, DEVICE)
        y_test, yhat_test = _predict(model, test_loader, target_scaler, DEVICE)
        y_gift, yhat_gift = _predict(model, gift_loader, target_scaler, DEVICE)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
        _plot_scatter(axes[0, 0], y_train, yhat_train, f"Fold {fold_id} - Train")
        _plot_scatter(axes[0, 1], y_val, yhat_val, f"Fold {fold_id} - Val")
        _plot_scatter(axes[1, 0], y_test, yhat_test, f"Fold {fold_id} - Test")
        _plot_scatter(axes[1, 1], y_gift, yhat_gift, f"Fold {fold_id} - GIFT")

        out_path = out_dir / f"fold_{fold_id}_predictions.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
