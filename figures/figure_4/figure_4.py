"""Plot EVA scale-binned NRMSE for the three MuScaRi feature variants."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muscari.muscari import MuScaRi
from muscari.utils import add_effort_columns


ROOT = Path(__file__).parents[2]
SBCV_DATASET_ID = "ceacce0"
SBCV_DATA_DIR = ROOT / "data/processed/training_samples/sbcv" / SBCV_DATASET_ID
ARTIFACT_ROOT = ROOT / "scripts/results/benchmark/artifacts"
OUTPUT_DIR = Path(__file__).parent
FIGURE_PATH = OUTPUT_DIR / "figure_4.pdf"
PAPER_FIGURE_PATH: Path | None = ROOT / "paper" / "figures" / "figure_4.pdf"
CSV_PATH = OUTPUT_DIR / "figure_4_normalized_rmse_by_area.csv"

DEVICE = os.environ.get("MUSCARI_FIGURE4_DEVICE", "cpu")
FOLD_IDS = range(5)
N_AREA_BINS = 20
MIN_AREA_KM2 = 4.0
MAX_AREA_KM2 = 1e6
LOG_AREA_EDGES = np.linspace(
    np.log(MIN_AREA_KM2 * 1e6),
    np.log(MAX_AREA_KM2 * 1e6),
    N_AREA_BINS + 1,
)

MODEL_SPECS = [
    {
        "name": "MuScaRi_Area",
        "label": "Area only",
        "config_hash": "ac733d9bd2f6",
        "color": "#f72585",
        "marker": "o",
    },
    {
        "name": "MuScaRi_ClimateDEM",
        "label": "Environment only",
        "config_hash": "dae0789a3c87",
        "color": "#3a0ca3",
        "marker": "^",
    },
    {
        "name": "MuScaRi_ClimateDEM_Area",
        "label": "Environment + area",
        "config_hash": "ad74a3020281",
        "color": "#4cc9f0",
        "marker": "s",
    },
]


def unique_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def load_fold_model(run_dir: Path, fold_id: int) -> tuple[MuScaRi, str]:
    checkpoint_path = run_dir / f"fold_{fold_id}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    return MuScaRi.initialize(checkpoint, device=DEVICE), config.effort_transform


def prepare_fold_data(path: Path, feature_names: list[str], effort_transform: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing EVA test fold: {path}")
    raw_features = [
        name
        for name in feature_names
        if name not in {"log_sp_unit_area", "log_observed_area"}
    ]
    columns = unique_columns(["sr", "sp_unit_area", "observed_area"] + raw_features)
    df = pd.read_parquet(path, columns=columns)
    df = add_effort_columns(df, effort_transform)
    df = df.replace([np.inf, -np.inf], np.nan)
    required = unique_columns(
        ["sr", "sp_unit_area", "observed_area", "log_sp_unit_area", "log_observed_area"]
        + feature_names
    )
    return df.dropna(subset=required).copy()


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape or not y_true.size:
        raise ValueError("Observed and predicted richness must have the same non-empty shape.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Observed and predicted richness must be finite.")
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mean_sr = float(np.mean(y_true))
    if not np.isfinite(mean_sr) or mean_sr <= 0:
        raise ValueError("Mean observed richness must be finite and positive.")
    return rmse, mean_sr, rmse / mean_sr


def evaluate_model_bins(spec: dict) -> list[dict]:
    run_dir = ARTIFACT_ROOT / spec["name"] / spec["config_hash"]
    rows = []
    for fold_id in FOLD_IDS:
        model, effort_transform = load_fold_model(run_dir, fold_id)
        fold_path = SBCV_DATA_DIR / f"fold_{fold_id}_test.parquet"
        df = prepare_fold_data(
            fold_path,
            model.feature_names,
            effort_transform=effort_transform,
        )
        df["prediction"] = model.predict_sr(df).reshape(-1)
        df["area_bin"] = pd.cut(
            df["log_sp_unit_area"],
            bins=LOG_AREA_EDGES,
            labels=False,
            include_lowest=True,
        )
        if df["area_bin"].isna().any():
            missing = int(df["area_bin"].isna().sum())
            raise ValueError(f"{missing} rows fell outside the configured area bins.")

        grouped = df.groupby("area_bin", sort=True, observed=False)
        for area_bin, group in grouped:
            y_true = group["sr"].to_numpy(dtype=float)
            y_pred = group["prediction"].to_numpy(dtype=float)
            rmse, mean_sr, nrmse = normalized_rmse(y_true, y_pred)
            bin_idx = int(area_bin)
            rows.append(
                {
                    "dataset_id": SBCV_DATASET_ID,
                    "model_name": spec["name"],
                    "model_label": spec["label"],
                    "config_hash": spec["config_hash"],
                    "fold": fold_id,
                    "area_bin": bin_idx,
                    "bin_left_km2": float(np.exp(LOG_AREA_EDGES[bin_idx]) / 1e6),
                    "bin_right_km2": float(np.exp(LOG_AREA_EDGES[bin_idx + 1]) / 1e6),
                    "area_center_km2": float(np.exp(group["log_sp_unit_area"].mean()) / 1e6),
                    "n_samples": int(len(group)),
                    "mean_sr": mean_sr,
                    "rmse": rmse,
                    "nrmse": nrmse,
                    "nrmse_percent": 100.0 * nrmse,
                }
            )
    return rows


def summarize_for_plot(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["model_name", "model_label", "area_bin"], sort=True)
    summary = grouped.agg(
        area_center_km2=("area_center_km2", "mean"),
        nrmse_percent_mean=("nrmse_percent", "mean"),
        nrmse_percent_std=("nrmse_percent", "std"),
    )
    return summary.reset_index()


def plot_results(results: pd.DataFrame) -> None:
    summary = summarize_for_plot(results)
    fig, ax = plt.subplots(figsize=(4, 4))

    for spec in MODEL_SPECS:
        data = summary[summary["model_name"] == spec["name"]].sort_values("area_bin")
        y = data["nrmse_percent_mean"].to_numpy(dtype=float)
        y_std = data["nrmse_percent_std"].to_numpy(dtype=float)
        x = data["area_center_km2"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker=spec["marker"],
            markersize=4,
            linestyle="-",
            color=spec["color"],
            label=spec["label"],
            alpha=0.85,
        )
        ax.fill_between(
            x,
            np.maximum(y - y_std, 1e-9),
            y + y_std,
            alpha=0.2,
            color=spec["color"],
        )

    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("NRMSE (%)")
    ax.set_xlabel(r"Spatial unit area, $A$ (km$^2$)")
    ax.legend(frameon=True, fancybox=True, bbox_to_anchor=(0.5, 1.2), loc="center")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    if PAPER_FIGURE_PATH is not None:
        fig.savefig(PAPER_FIGURE_PATH, dpi=300, bbox_inches="tight")


def validate_results(results: pd.DataFrame) -> None:
    expected_rows = len(MODEL_SPECS) * len(list(FOLD_IDS)) * N_AREA_BINS
    if len(results) != expected_rows:
        raise ValueError(f"Expected {expected_rows} result rows, found {len(results)}.")
    if results["nrmse_percent"].isna().any():
        raise ValueError("NRMSE contains missing values.")
    if (results["n_samples"] <= 0).any():
        raise ValueError("At least one model/fold/area bin has no samples.")


def main() -> None:
    rows = []
    for spec in MODEL_SPECS:
        rows.extend(evaluate_model_bins(spec))

    results = pd.DataFrame(rows)
    validate_results(results)
    results.to_csv(CSV_PATH, index=False)
    plot_results(results)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
