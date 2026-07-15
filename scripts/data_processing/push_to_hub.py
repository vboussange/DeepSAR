"""Publish MuScaRi datasets and the selected pretrained ensemble."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

from muscari import MuScaRiEnsemble
from muscari.data_processing.utils_eva import EVADataset
from muscari.data_processing.utils_features import EnvironmentalFeatureDataset
from muscari.data_processing.utils_gift import GIFTDataset


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_REPO = "vboussange/muscari"
DATASET_REPO = "vboussange/muscari-data"
EXPORT_DIR = (
    ROOT_DIR
    / "scripts/results/benchmark/artifacts/MuScaRi_ClimateDEM"
    / "dae0789a3c87"
    / "ensemble_pretrained"
)
MODEL_CARD_PATH = ROOT_DIR / "muscari/MODEL_CARD.md"
DATASET_CARD_PATH = ROOT_DIR / "muscari/data_processing/DATASET_CARD.md"
SBCV_DIR = ROOT_DIR / "data/processed/training_samples/sbcv/ceacce0"
GIFT_DIR = ROOT_DIR / "data/processed/test_samples_GIFT/418c563"
MODEL_FILES = ("README.md", "config.json", "metadata.json", "model.safetensors")


def validate_model_export() -> MuScaRiEnsemble:
    required = ("config.json", "metadata.json", "model.safetensors")
    for path in (MODEL_CARD_PATH, *(EXPORT_DIR / name for name in required)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing release file: {path}")

    config = json.loads((EXPORT_DIR / "config.json").read_text())
    metadata = json.loads((EXPORT_DIR / "metadata.json").read_text())
    export = metadata.get("export", {})
    features = metadata.get("features_and_labels", {}).get("feature_columns")
    if export.get("config_hash") != EXPORT_DIR.parent.name:
        raise ValueError("Model path and metadata configuration hash differ")
    if export.get("n_models") != config.get("n_models"):
        raise ValueError("Config and metadata ensemble sizes differ")
    if features != config.get("feature_names"):
        raise ValueError("Config and metadata feature order differ")

    model = MuScaRiEnsemble.from_pretrained(EXPORT_DIR)
    if model.n_models != config.get("n_models"):
        raise ValueError("Loaded ensemble size differs from its config")
    if list(model.feature_names) != config.get("feature_names"):
        raise ValueError("Loaded ensemble feature order differs from its config")
    if not np.allclose(model.ensemble_weights, export.get("ensemble_weights")):
        raise ValueError("Loaded ensemble weights differ from its metadata")
    return model


def prediction_probe(model: MuScaRiEnsemble) -> np.ndarray:
    feature_names = list(model.feature_names)
    frame = pd.DataFrame(
        [np.zeros(len(feature_names)), np.ones(len(feature_names))],
        columns=feature_names,
    )
    predictions = np.atleast_1d(model.predict_mean_sr_tot(frame)).astype(float)
    if not np.isfinite(predictions).all() or not (predictions > 0).all():
        raise ValueError("Model prediction probe produced invalid values")
    return predictions


def generated_dataset_files() -> dict[str, Path]:
    sbcv_partitions = sorted(SBCV_DIR.glob("fold_*.parquet"))
    sbcv_summaries = sorted(SBCV_DIR.glob("fold_*_summary.json"))
    if not sbcv_partitions:
        raise FileNotFoundError(f"No spatial cross-validation partitions in {SBCV_DIR}")
    partition_stems = {path.stem for path in sbcv_partitions}
    summary_stems = {path.name.removesuffix("_summary.json") for path in sbcv_summaries}
    if partition_stems != summary_stems:
        raise ValueError("Spatial cross-validation partitions and summaries differ")

    gift_files = sorted(GIFT_DIR.glob("*.parquet")) + sorted(
        GIFT_DIR.glob("*_summary.json")
    )
    if not gift_files:
        raise FileNotFoundError(f"No total-richness extrapolation samples in {GIFT_DIR}")

    files = {
        f"generated_samples/sbcv/{SBCV_DIR.name}/{path.name}": path
        for path in sbcv_partitions + sbcv_summaries
    }
    files.update(
        {f"generated_samples/GIFT/{GIFT_DIR.name}/{path.name}": path for path in gift_files}
    )
    return files


def upload_folder(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    source_dir: Path,
    parent_commit: str,
    commit_message: str,
    path_in_repo: str | None = None,
    allow_patterns: list[str] | None = None,
):
    return api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        revision="main",
        folder_path=source_dir,
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        parent_commit=parent_commit,
        commit_message=commit_message,
    )


def verify_remote_datasets(api: HfApi, revision: str, expected: dict[str, Path]) -> None:
    info = api.dataset_info(DATASET_REPO, revision=revision, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    for path_in_repo, local_path in expected.items():
        if path_in_repo not in remote:
            raise ValueError(f"Remote dataset is missing {path_in_repo}")
        if remote[path_in_repo].size != local_path.stat().st_size:
            raise ValueError(f"Remote dataset size differs for {path_in_repo}")


def verify_remote_model(
    revision: str,
    local_model: MuScaRiEnsemble,
    local_predictions: np.ndarray,
) -> None:
    with tempfile.TemporaryDirectory(prefix="muscari-hf-verify-") as tmp:
        remote_dir = Path(tmp) / "model"
        snapshot_download(
            repo_id=MODEL_REPO,
            repo_type="model",
            revision=revision,
            allow_patterns=list(MODEL_FILES),
            local_dir=remote_dir,
        )
        remote_model = MuScaRiEnsemble.from_pretrained(remote_dir)
        if remote_model.n_models != local_model.n_models:
            raise ValueError("Remote and local ensemble sizes differ")
        if list(remote_model.feature_names) != list(local_model.feature_names):
            raise ValueError("Remote and local feature order differs")
        if not np.allclose(remote_model.ensemble_weights, local_model.ensemble_weights):
            raise ValueError("Remote and local ensemble weights differ")
        if not np.allclose(prediction_probe(remote_model), local_predictions, atol=1e-6):
            raise ValueError("Remote and local prediction probes differ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Publish after validation. Without this flag, perform a dry run.",
    )
    parser.add_argument(
        "--refresh-source-data",
        action="store_true",
        help="Rebuild and upload EVA, GIFT, and environmental source data.",
    )
    parser.add_argument("--expected-model-parent")
    parser.add_argument("--expected-dataset-parent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_source_data and not args.upload:
        raise SystemExit("--refresh-source-data requires --upload")
    if args.upload and not (args.expected_model_parent and args.expected_dataset_parent):
        raise SystemExit(
            "--upload requires --expected-model-parent and --expected-dataset-parent"
        )

    model = validate_model_export()
    predictions = prediction_probe(model)
    data_files = generated_dataset_files()
    api = HfApi()
    model_info = api.model_info(MODEL_REPO, revision="main")
    dataset_info = api.dataset_info(DATASET_REPO, revision="main")
    if args.expected_model_parent and model_info.sha != args.expected_model_parent:
        raise RuntimeError(f"Model main advanced to {model_info.sha}")
    if args.expected_dataset_parent and dataset_info.sha != args.expected_dataset_parent:
        raise RuntimeError(f"Dataset main advanced to {dataset_info.sha}")

    data_size = sum(path.stat().st_size for path in data_files.values())
    print(f"Model main: {model_info.sha}")
    print(f"Dataset main: {dataset_info.sha}")
    print(f"Model files: {', '.join(MODEL_FILES)}")
    print(f"Generated datasets: {len(data_files)} files, {data_size / 1024**2:.1f} MiB")
    print(f"Prediction probe: {predictions.tolist()}")
    print(f"Refresh source data: {args.refresh_source_data}")
    print("No remote files will be deleted.")
    if not args.upload:
        print("Dry run complete; no Hub changes made.")
        print(
            "Upload with: uv run python scripts/data_processing/push_to_hub.py --upload "
            f"--expected-model-parent {model_info.sha} "
            f"--expected-dataset-parent {dataset_info.sha}"
        )
        return

    account = api.whoami()
    print(f"Authenticated Hugging Face account: {account['name']}")
    if args.refresh_source_data:
        EVADataset().push_to_hub(DATASET_REPO)
        GIFTDataset().push_to_hub(DATASET_REPO)
        EnvironmentalFeatureDataset().push_to_hub(DATASET_REPO)
        dataset_info = api.dataset_info(DATASET_REPO, revision="main")

    with tempfile.TemporaryDirectory(prefix="muscari-hf-release-") as tmp:
        tmp_dir = Path(tmp)
        model_dir = tmp_dir / "model"
        dataset_card_dir = tmp_dir / "dataset_card"
        model_dir.mkdir()
        dataset_card_dir.mkdir()
        shutil.copyfile(MODEL_CARD_PATH, model_dir / "README.md")
        shutil.copyfile(DATASET_CARD_PATH, dataset_card_dir / "README.md")
        for filename in MODEL_FILES[1:]:
            shutil.copyfile(EXPORT_DIR / filename, model_dir / filename)

        dataset_commit = upload_folder(
            api,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            source_dir=SBCV_DIR,
            path_in_repo=f"generated_samples/sbcv/{SBCV_DIR.name}",
            allow_patterns=["fold_*.parquet", "fold_*_summary.json"],
            parent_commit=dataset_info.sha,
            commit_message="Upload EVA spatial cross-validation samples",
        )
        dataset_commit = upload_folder(
            api,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            source_dir=GIFT_DIR,
            path_in_repo=f"generated_samples/GIFT/{GIFT_DIR.name}",
            allow_patterns=["*.parquet", "*_summary.json"],
            parent_commit=dataset_commit.oid,
            commit_message="Upload GIFT total-richness extrapolation samples",
        )
        dataset_commit = upload_folder(
            api,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            source_dir=dataset_card_dir,
            parent_commit=dataset_commit.oid,
            commit_message="Document generated MuScaRi samples",
        )
        verify_remote_datasets(api, dataset_commit.oid, data_files)

        model_commit = upload_folder(
            api,
            repo_id=MODEL_REPO,
            repo_type="model",
            source_dir=model_dir,
            parent_commit=model_info.sha,
            commit_message="Publish MuScaRi ClimateDEM ensemble",
        )
        verify_remote_model(model_commit.oid, model, predictions)

    print(f"Dataset commit: {dataset_commit.commit_url}")
    print(f"Model commit: {model_commit.commit_url}")
    print("Remote datasets and model verified.")


if __name__ == "__main__":
    main()
