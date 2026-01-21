"""Generate LaTeX table with sample counts per fold (train/val/test)."""
from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).parents[3]
SBCV_SAMPLES_PATH = ROOT / "data" / "processed" / "training_samples" / "sbcv" / "606e055"
OUTPUT_TEX = Path(__file__).with_name("fold_sample_counts.tex")


def load_fold_counts(sbcv_path: Path) -> pd.DataFrame:
    rows = []
    for fold_id in range(5):
        train_path = sbcv_path / f"fold_{fold_id}_train.parquet"
        val_path = sbcv_path / f"fold_{fold_id}_val.parquet"
        test_path = sbcv_path / f"fold_{fold_id}_test.parquet"

        if not (train_path.exists() and val_path.exists() and test_path.exists()):
            continue

        n_train = len(gpd.read_parquet(train_path))
        n_val = len(gpd.read_parquet(val_path))
        n_test = len(gpd.read_parquet(test_path))

        rows.append(
            {
                "Fold": fold_id,
                "Train": n_train,
                "Validation": n_val,
                "Test": n_test,
            }
        )

    return pd.DataFrame(rows)


def render_latex_table(df: pd.DataFrame) -> str:
    header = (
        "\\begin{table}\n"
        "    \\centering\n"
        "    \\small\n"
        "    \\setlength{\\tabcolsep}{6pt}\n"
        "    \\begin{tabularx}{\\textwidth}{l >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X}\n"
        "    \\toprule\n"
        "    Fold & Train & Validation & Test \\\\\n"
        "    \\midrule\n"
    )
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"    {int(row['Fold'])} & {int(row['Train'])} & {int(row['Validation'])} & {int(row['Test'])} \\\\\n"
        )
    footer = (
        "    \\bottomrule\n"
        "    \\end{tabularx}\n"
        "    \\caption{Sample counts per fold for training, validation, and test datasets.}\n"
        "    \\label{tab:fold_sample_counts}\n"
        "\\end{table}\n"
    )
    return header + "".join(rows) + footer




if __name__ == "__main__":
    df = load_fold_counts(SBCV_SAMPLES_PATH)
    latex = render_latex_table(df)
    OUTPUT_TEX.write_text(latex)
    print(latex)
