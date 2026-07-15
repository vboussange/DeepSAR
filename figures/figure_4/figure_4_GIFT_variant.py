"""GIFT variant of Figure 4 with quantile-binned normalized RMSE by area."""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_4 as base


GIFT_DATASET_ID = "418c563"
GIFT_DATA_PATH = (
    base.ROOT / "data/processed/test_samples_GIFT" / GIFT_DATASET_ID / "compiled_data.parquet"
)
FIGURE_PATH = base.OUTPUT_DIR / "figure_4_GIFT_variant.pdf"
CSV_PATH = base.OUTPUT_DIR / "figure_4_GIFT_variant_normalized_rmse_by_area.csv"
base.PAPER_FIGURE_PATH = None
N_GIFT_BINS = 5
MIN_BIN_SAMPLES = 30

MODEL_SPECS = base.MODEL_SPECS + [
    {
        "name": "MuScaRi_ClimateDEM_Area",
        "label": "Climate + DEM + area",
        "config_hash": "ad74a3020281",
        "color": "#4cc9f0",
        "marker": "s",
    },
]


def run_dir_for(spec: dict) -> base.Path:
    return base.ARTIFACT_ROOT / spec["name"] / spec["config_hash"]


def load_feature_names_by_model() -> dict[str, list[str]]:
    feature_names = {}
    for spec in MODEL_SPECS:
        config_path = run_dir_for(spec) / "config.json"
        with open(config_path) as handle:
            config = json.load(handle)
        feature_names[spec["name"]] = config["features_and_labels"]["feature_columns"]
    return feature_names


def prepare_gift_data(feature_names_by_model: dict[str, list[str]]) -> pd.DataFrame:
    raw_features = []
    for feature_names in feature_names_by_model.values():
        raw_features.extend(
            name
            for name in feature_names
            if name not in {"log_sp_unit_area", "log_observed_area"}
        )
    columns = base.unique_columns(["sr", "sp_unit_area", "observed_area"] + raw_features)
    df = pd.read_parquet(GIFT_DATA_PATH, columns=columns)
    df = df.copy()
    df["observed_area"] = df["sp_unit_area"]
    df = base.add_effort_columns(df, "absolute")
    df = df.replace([np.inf, -np.inf], np.nan)
    required = base.unique_columns(
        ["sr", "sp_unit_area", "observed_area", "log_sp_unit_area", "log_observed_area"]
        + raw_features
    )
    df = df.dropna(subset=required).copy()
    df["area_bin"] = pd.qcut(
        df["log_sp_unit_area"],
        q=N_GIFT_BINS,
        labels=False,
        duplicates="raise",
    )
    counts = df["area_bin"].value_counts()
    if counts.min() < MIN_BIN_SAMPLES:
        raise ValueError(
            f"GIFT binning produced a bin with only {counts.min()} samples; "
            f"minimum required is {MIN_BIN_SAMPLES}."
        )
    return df


def evaluate_gift_bins() -> pd.DataFrame:
    feature_names_by_model = load_feature_names_by_model()
    gift_df = prepare_gift_data(feature_names_by_model)
    rows = []

    for spec in MODEL_SPECS:
        run_dir = run_dir_for(spec)
        for fold_id in base.FOLD_IDS:
            model, _ = base.load_fold_model(run_dir, fold_id)
            df = gift_df.dropna(subset=feature_names_by_model[spec["name"]]).copy()
            df["prediction"] = model.predict_sr_tot(df).reshape(-1)

            grouped = df.groupby("area_bin", sort=True, observed=False)
            for area_bin, group in grouped:
                y_true = group["sr"].to_numpy(dtype=float)
                y_pred = group["prediction"].to_numpy(dtype=float)
                rmse, mean_sr, nrmse = base.normalized_rmse(y_true, y_pred)
                rows.append(
                    {
                        "dataset_id": base.SBCV_DATASET_ID,
                        "gift_dataset_id": GIFT_DATASET_ID,
                        "model_name": spec["name"],
                        "model_label": spec["label"],
                        "config_hash": spec["config_hash"],
                        "fold": int(fold_id),
                        "area_bin": int(area_bin),
                        "bin_left_km2": float(group["sp_unit_area"].min() / 1e6),
                        "bin_right_km2": float(group["sp_unit_area"].max() / 1e6),
                        "area_center_km2": float(np.exp(group["log_sp_unit_area"].mean()) / 1e6),
                        "n_samples": int(len(group)),
                        "mean_sr": mean_sr,
                        "rmse": rmse,
                        "nrmse": nrmse,
                        "nrmse_percent": 100.0 * nrmse,
                    }
                )
    return pd.DataFrame(rows)


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
        x = data["area_center_km2"].to_numpy(dtype=float)
        y = data["nrmse_percent_mean"].to_numpy(dtype=float)
        y_std = data["nrmse_percent_std"].to_numpy(dtype=float)
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
            np.maximum(y - y_std, 0.0),
            y + y_std,
            alpha=0.2,
            color=spec["color"],
        )

    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("NRMSE (%)")
    ax.set_xlabel(r"GIFT area, $A$ (km$^2$)")
    ax.legend(frameon=True, fancybox=True, bbox_to_anchor=(0.5, 1.2), loc="center")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")


def validate_results(results: pd.DataFrame) -> None:
    expected_rows = len(MODEL_SPECS) * len(list(base.FOLD_IDS)) * N_GIFT_BINS
    if len(results) != expected_rows:
        raise ValueError(f"Expected {expected_rows} result rows, found {len(results)}.")
    if results["nrmse_percent"].isna().any():
        raise ValueError("NRMSE contains missing values.")
    if results.groupby("area_bin")["n_samples"].first().min() < MIN_BIN_SAMPLES:
        raise ValueError("At least one GIFT bin has too few samples.")


def main() -> None:
    results = evaluate_gift_bins()
    validate_results(results)
    results.to_csv(CSV_PATH, index=False)
    plot_results(results)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
