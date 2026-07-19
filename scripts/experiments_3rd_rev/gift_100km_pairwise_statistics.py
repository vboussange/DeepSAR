"""Calculate fold-paired statistics for the 100 km GIFT comparison."""

from itertools import combinations, product
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).parents[2]
DATASET_ID = "d0848f6"
CANONICAL_GIFT_DATASET_ID = "418c563"
SOURCE_GIFT_DATASET_ID = "da569da"
RESULTS_DIR = ROOT / "scripts" / "results" / "gift_asymptote_evaluation" / DATASET_ID
INPUT_PATH = RESULTS_DIR / "gift_asymptote_evaluation_results.csv"
OUTPUT_CSV = RESULTS_DIR / "gift_100km_pairwise_statistics.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "gift_100km_pairwise_statistics.md"

MODEL_LABELS = {
    "MuScaRi_Area": "Area only",
    "MuScaRi_ClimateDEM": "Environment only",
    "MuScaRi_ClimateDEM_Area": "Environment and area",
}
MODEL_ORDER = list(MODEL_LABELS)
METRICS = {
    "nrmse_percent": ("nrmse", lambda values: 100.0 * values),
    "absolute_median_relative_bias": ("median_relative_bias", np.abs),
}
ALPHA = 0.05


def verify_gift_equivalence() -> None:
    gift_root = ROOT / "data" / "processed" / "test_samples_GIFT"
    source = gpd.read_parquet(gift_root / SOURCE_GIFT_DATASET_ID / "compiled_data.parquet")
    canonical = gpd.read_parquet(
        gift_root / CANONICAL_GIFT_DATASET_ID / "compiled_data.parquet"
    )
    if list(source.columns) != list(canonical.columns) or source.crs != canonical.crs:
        raise AssertionError("Source and canonical GIFT schemas differ")
    non_geometry = [column for column in source.columns if column != source.geometry.name]
    pd.testing.assert_frame_equal(
        source[non_geometry], canonical[non_geometry], check_dtype=False, check_exact=True
    )
    if not source.geometry.geom_equals_exact(canonical.geometry, tolerance=0).all():
        raise AssertionError("Source and canonical GIFT geometries differ")


def load_fold_results() -> pd.DataFrame:
    results = pd.read_csv(INPUT_PATH)
    folds = results[
        (results["prediction_mode"] == "asymptotic_total")
        & (results["aggregation"] == "fold_member")
        & (results["model_name"].isin(MODEL_ORDER))
    ].copy()
    folds["fold"] = folds["fold"].astype(int)

    expected_pairs = pd.MultiIndex.from_product(
        [MODEL_ORDER, range(5)], names=["model_name", "fold"]
    )
    observed_pairs = pd.MultiIndex.from_frame(folds[["model_name", "fold"]])
    if len(folds) != 15 or not observed_pairs.is_unique or set(observed_pairs) != set(expected_pairs):
        raise AssertionError("Expected exactly the same five folds for all three models")
    if folds["n_samples"].nunique() != 1 or int(folds["n_samples"].iloc[0]) != 178:
        raise AssertionError("Models were not evaluated on the same 178-sample GIFT cohort")
    for column in ["dataset_id", "sr_mean", "sr_median"]:
        if folds[column].nunique() != 1:
            raise AssertionError(f"Fold rows disagree on {column}")
    if folds["dataset_id"].iloc[0] != DATASET_ID:
        raise AssertionError("Unexpected spatial-block dataset")
    return folds.sort_values(["model_name", "fold"])


def t_confidence_interval(differences: np.ndarray) -> tuple[float, float]:
    sem = stats.sem(differences)
    low, high = stats.t.interval(
        confidence=1.0 - ALPHA,
        df=len(differences) - 1,
        loc=np.mean(differences),
        scale=sem,
    )
    return float(low), float(high)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    permuted = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in product([-1.0, 1.0], repeat=len(differences))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    order = np.argsort(pvalues.to_numpy())
    adjusted = np.empty(len(pvalues), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (len(pvalues) - rank) * float(pvalues.iloc[index])
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)
    return pd.Series(adjusted, index=pvalues.index)


def calculate_pairwise_statistics(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, (source_column, transform) in METRICS.items():
        values = {
            model: transform(
                folds[folds["model_name"] == model]
                .sort_values("fold")[source_column]
                .to_numpy(dtype=float)
            )
            for model in MODEL_ORDER
        }
        for model_a, model_b in combinations(MODEL_ORDER, 2):
            values_a = values[model_a]
            values_b = values[model_b]
            differences = values_b - values_a
            ci_low, ci_high = t_confidence_interval(differences)
            raw_p = float(stats.ttest_rel(values_b, values_a).pvalue)
            rows.append(
                {
                    "metric": metric,
                    "model_a": model_a,
                    "model_a_label": MODEL_LABELS[model_a],
                    "model_b": model_b,
                    "model_b_label": MODEL_LABELS[model_b],
                    "difference_definition": "model_b_minus_model_a",
                    "folds": "0;1;2;3;4",
                    "model_a_fold_values": ";".join(f"{value:.9g}" for value in values_a),
                    "model_b_fold_values": ";".join(f"{value:.9g}" for value in values_b),
                    "paired_fold_differences": ";".join(
                        f"{value:.9g}" for value in differences
                    ),
                    "model_a_fold_mean": float(np.mean(values_a)),
                    "model_b_fold_mean": float(np.mean(values_b)),
                    "mean_paired_difference": float(np.mean(differences)),
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "paired_t_p_raw": raw_p,
                    "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                }
            )
    pairwise = pd.DataFrame(rows)
    pairwise["paired_t_p_holm"] = pairwise.groupby("metric", group_keys=False)[
        "paired_t_p_raw"
    ].transform(holm_adjust)
    pairwise["holm_significant_0_05"] = pairwise["paired_t_p_holm"] < ALPHA
    pairwise["sign_flip_significant_0_05"] = pairwise["exact_sign_flip_p"] < ALPHA
    pairwise["qualitative_test_disagreement"] = (
        pairwise["holm_significant_0_05"] != pairwise["sign_flip_significant_0_05"]
    )
    return pairwise


def format_fold_values(values: str, digits: int) -> str:
    return ", ".join(f"{float(value):.{digits}f}" for value in values.split(";"))


def write_summary(pairwise: pd.DataFrame, folds: pd.DataFrame) -> None:
    target = pairwise[
        (pairwise["model_a"] == "MuScaRi_ClimateDEM")
        & (pairwise["model_b"] == "MuScaRi_ClimateDEM_Area")
    ].set_index("metric")
    nrmse = target.loc["nrmse_percent"]
    bias = target.loc["absolute_median_relative_bias"]

    source_ids = ", ".join(sorted(folds["gift_dataset_id"].unique()))
    lines = [
        "# Fold-paired 100 km GIFT statistics",
        "",
        f"Source: `{INPUT_PATH.relative_to(ROOT)}` (`asymptotic_total`, `fold_member` rows).",
        f"All models use folds 0--4 and the same {int(folds['n_samples'].iloc[0])} GIFT samples. "
        f"The results file records GIFT dataset `{source_ids}`; its compiled data are exactly "
        f"equivalent in columns, values, CRS, and geometry to canonical dataset "
        f"`{CANONICAL_GIFT_DATASET_ID}`.",
        "",
        "## Environment and area versus environment only",
        "",
        f"Fold NRMSE values (%) were {format_fold_values(nrmse.model_a_fold_values, 2)} "
        f"for environment only and {format_fold_values(nrmse.model_b_fold_values, 2)} for "
        f"environment and area. Their fold means were {nrmse.model_a_fold_mean:.2f}% and "
        f"{nrmse.model_b_fold_mean:.2f}%, respectively. The mean paired difference "
        f"(environment and area minus environment only) was {nrmse.mean_paired_difference:.2f} "
        f"percentage points (95% CI [{nrmse.ci_95_low:.2f}, {nrmse.ci_95_high:.2f}]; "
        f"raw paired t-test P={nrmse.paired_t_p_raw:.3f}; Holm-adjusted "
        f"P={nrmse.paired_t_p_holm:.3f}). The exact sign-flip P value was "
        f"{nrmse.exact_sign_flip_p:.4f}.",
        "",
        f"Fold-level absolute median relative biases were "
        f"{format_fold_values(bias.model_a_fold_values, 3)} for environment only and "
        f"{format_fold_values(bias.model_b_fold_values, 3)} for environment and area. "
        f"The mean paired difference was {bias.mean_paired_difference:.3f} "
        f"(95% CI [{bias.ci_95_low:.3f}, {bias.ci_95_high:.3f}]; raw paired t-test "
        f"P={bias.paired_t_p_raw:.3f}; Holm-adjusted P={bias.paired_t_p_holm:.3f}). "
        f"The exact sign-flip P value was {bias.exact_sign_flip_p:.4f}.",
        "",
        "The Holm-adjusted paired tests and exact sign-flip sensitivity checks agree "
        "qualitatively at alpha=0.05 for both targeted contrasts. The combined model has "
        "lower aggregate NRMSE and lower absolute fold-level median relative bias, but neither "
        "contrast meets the adjusted significance threshold with five folds.",
        "",
        "## Full pairwise results",
        "",
        f"See `{OUTPUT_CSV.relative_to(ROOT)}` for all three pairwise comparisons within each "
        "metric-specific Holm family, including fold values and exact sign-flip results.",
        "",
    ]
    OUTPUT_SUMMARY.write_text("\n".join(lines))


def main() -> None:
    verify_gift_equivalence()
    folds = load_fold_results()
    pairwise = calculate_pairwise_statistics(folds)
    pairwise.to_csv(OUTPUT_CSV, index=False)
    write_summary(pairwise, folds)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
