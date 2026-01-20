import matplotlib.pyplot as plt
import pickle
import logging
from pathlib import Path
import torch
import torch.nn as nn
import git

from deepsar.deep4pweibull import Deep4PWeibull
from deepsar.ensemble_model import DeepSAREnsembleModel

class MSELogLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELogLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, input, target):
        log_input = torch.log(torch.clamp(input, min=1e-8))
        log_target = torch.log(torch.clamp(target, min=1e-8))
        loss = (log_input - log_target) ** 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
        
def save_to_pickle(filepath, **kwargs):
    objects_dict = kwargs
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as file:
        pickle.dump(objects_dict, file)
    logging.info(f"Results saved at {filepath}")

def symmetric_arch(n, base=32, factor=2):
    half = (n + 1) // 2
    front = [base * factor**i for i in range(half)]
    mirror = front[:-1] if n % 2 else front
    return front + mirror[::-1]


def get_git_hash(short=True, fallback="unknown"):
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.git.rev_parse(repo.head, short=short)
    except git.InvalidGitRepositoryError:
        logging.warning("Could not determine git hash; using '%s'.", fallback)
        return fallback


def load_ensemble_from_folds(run_dir: Path, device: str = "cpu", return_config: bool = False):
    ckpt_paths = sorted(run_dir.glob("fold_*.pth"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No fold_*.pth files found in {run_dir}")

    models = []
    feature_names_ref = None
    config_ref = None
    for ckpt_path in ckpt_paths:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        feature_names = checkpoint["feature_names"]
        if feature_names_ref is None:
            feature_names_ref = feature_names
            config_ref = checkpoint.get("config")
        else:
            assert feature_names_ref == feature_names, "Feature names differ across folds"

        config = checkpoint["config"]
        model = Deep4PWeibull(
            config.layer_sizes,
            feature_names=feature_names,
            feature_scaler=checkpoint["feature_scaler"],
            target_scaler=checkpoint["target_scaler"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    ensemble = DeepSAREnsembleModel(models)
    ensemble.eval()
    if return_config:
        return ensemble, config_ref
    return ensemble