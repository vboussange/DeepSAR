"""
Push all MuScaRi datasets and the pretrained model to the Hugging Face Hub.

Edit the constants below, then run::

    python scripts/push_to_hub.py

All three dataset caches and the model ensemble are uploaded to the same
repositories using the sub-folder layout::

    DATASET_REPO/
        EVA/species_matrix.parquet
        GIFT/species_matrix.parquet
        environmental_features/chelsa_dem_cache.nc
                               landcover_cache.nc

    MODEL_REPO/
        config.json          ← written by PyTorchModelHubMixin
        model.safetensors    ← written by PyTorchModelHubMixin
"""

from pathlib import Path

from muscari.data_processing.utils_eva import EVADataset
from muscari.data_processing.utils_gift import GIFTDataset
from muscari.data_processing.utils_features import EnvironmentalFeatureDataset
from muscari.ensemble_model import MuScaRiEnsemble

DATASET_REPO = "vboussange/muscari-data"
MODEL_REPO   = "vboussange/muscari"
RUN_DIR      = Path(__file__).parent / "results/train/ceacce0"

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Pushing EVA species/plot matrix …")
    print("=" * 60)
    EVADataset().push_to_hub(DATASET_REPO)

    print("\n" + "=" * 60)
    print("Pushing GIFT species/plot matrix …")
    print("=" * 60)
    GIFTDataset().push_to_hub(DATASET_REPO)

    print("\n" + "=" * 60)
    print("Pushing environmental feature caches …")
    print("=" * 60)
    EnvironmentalFeatureDataset().push_to_hub(DATASET_REPO)

    print("\n" + "=" * 60)
    print(f"Building ensemble from {RUN_DIR} …")
    print("=" * 60)
    ensemble = MuScaRiEnsemble.from_folds(RUN_DIR)
    print(f"Pushing ensemble ({ensemble.n_models} models) to {MODEL_REPO} …")
    ensemble.push_to_hub(MODEL_REPO)

    print("\n✓ All assets pushed successfully.")
    print(f"  Dataset : https://huggingface.co/datasets/{DATASET_REPO}")
    print(f"  Model   : https://huggingface.co/{MODEL_REPO}")
