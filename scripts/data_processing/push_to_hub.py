"""
Push all MuScaRi datasets and the pretrained model to the Hugging Face Hub.
"""

from pathlib import Path

from muscari.data_processing.utils_eva import EVADataset
from muscari.data_processing.utils_gift import GIFTDataset
from muscari.data_processing.utils_features import EnvironmentalFeatureDataset
from muscari import MuScaRiEnsemble

DATASET_REPO = "vboussange/muscari-data"
MODEL_REPO   = "vboussange/muscari"
RUN_DIR      = Path(__file__).parents[1] / "results/train/ceacce0"

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

    print("\nAll assets pushed successfully.")
    print(f"  Dataset : https://huggingface.co/datasets/{DATASET_REPO}")
    print(f"  Model   : https://huggingface.co/{MODEL_REPO}")
