from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin

from muscari.muscari import MuScaRi


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def _scaler_to_dict(scaler) -> dict:
    """Serialize a fitted sklearn scaler to a JSON-compatible dict."""
    cls_name = type(scaler).__name__
    d = {"cls": cls_name, "n_features_in_": int(scaler.n_features_in_)}
    if cls_name == "MinMaxScaler":
        d["min_"] = scaler.min_.tolist()
        d["scale_"] = scaler.scale_.tolist()
        d["data_min_"] = scaler.data_min_.tolist()
        d["data_max_"] = scaler.data_max_.tolist()
        d["data_range_"] = scaler.data_range_.tolist()
    elif cls_name == "MaxAbsScaler":
        d["scale_"] = scaler.scale_.tolist()
        d["max_abs_"] = scaler.max_abs_.tolist()
    elif cls_name == "StandardScaler":
        d["mean_"] = scaler.mean_.tolist()
        d["scale_"] = scaler.scale_.tolist()
    else:
        raise TypeError(f"Unsupported scaler type: {cls_name}")
    return d


def _dict_to_scaler(d: dict):
    """Reconstruct a scaler from a serialized dict."""
    from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
    cls_name = d["cls"]
    if cls_name == "MinMaxScaler":
        scaler = MinMaxScaler()
        scaler.min_ = np.array(d["min_"])
        scaler.scale_ = np.array(d["scale_"])
        scaler.data_min_ = np.array(d["data_min_"])
        scaler.data_max_ = np.array(d["data_max_"])
        scaler.data_range_ = np.array(d["data_range_"])
    elif cls_name == "MaxAbsScaler":
        scaler = MaxAbsScaler()
        scaler.scale_ = np.array(d["scale_"])
        scaler.max_abs_ = np.array(d["max_abs_"])
    elif cls_name == "StandardScaler":
        scaler = StandardScaler()
        scaler.mean_ = np.array(d["mean_"])
        scaler.scale_ = np.array(d["scale_"])
    else:
        raise TypeError(f"Unsupported scaler type: {cls_name}")
    scaler.n_features_in_ = d["n_features_in_"]
    return scaler


# ---------------------------------------------------------------------------
# MuScaRiEnsemble
# ---------------------------------------------------------------------------

class MuScaRiEnsemble(nn.Module, PyTorchModelHubMixin, library_name="muscari"):
    """
    Ensemble of :class:`MuScaRi` models.

    Construction
    ------------
    Preferred: use :meth:`from_models` to build from a list of trained
    :class:`MuScaRi` instances.

    Direct instantiation takes JSON-serializable arguments so that
    ``config.json`` on the Hub stays human-readable::

        ensemble = MuScaRiEnsemble(
            n_models=5,
            layer_sizes=[128, 64, 32, 64, 128],
            feature_names=["bio1", "bio12", ...],
            feature_scalers=[{"mean_": [...], "scale_": [...], "n_features_in_": 20}, ...],
            target_scalers=[...],
        )
    """

    def __init__(
        self,
        n_models: int,
        layer_sizes: list,
        feature_names: list,
        feature_scalers: Optional[list] = None,
        target_scalers: Optional[list] = None,
    ):
        super().__init__()
        self.n_models = n_models
        self.layer_sizes = layer_sizes
        self.feature_names = feature_names
        # Stored as list-of-dicts so config.json stays JSON-serializable
        self.feature_scalers = feature_scalers
        self.target_scalers = target_scalers

        # Convert dicts → sklearn scalers for the sub-models
        _fscalers = (
            [_dict_to_scaler(d) if d is not None else None for d in feature_scalers]
            if feature_scalers else [None] * n_models
        )
        _tscalers = (
            [_dict_to_scaler(d) if d is not None else None for d in target_scalers]
            if target_scalers else [None] * n_models
        )
        self.models = nn.ModuleList([
            MuScaRi(layer_sizes, feature_names, _fscalers[i], _tscalers[i])
            for i in range(n_models)
        ])

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_models(cls, models: list[MuScaRi]) -> "MuScaRiEnsemble":
        """Build an ensemble from a list of already-trained :class:`MuScaRi` models."""
        assert models, "models list must not be empty"
        m0 = models[0]

        # Infer layer_sizes from the first model's architecture
        layer_sizes = [block.linear.out_features for block in m0.ffnn.fully_connected_layers]

        def _to_dict(scaler):
            return _scaler_to_dict(scaler) if scaler is not None else None

        feature_scalers = [_to_dict(m.feature_scaler) for m in models]
        target_scalers  = [_to_dict(m.target_scaler)  for m in models]

        # Use None when all scalers are absent
        if all(s is None for s in feature_scalers):
            feature_scalers = None
        if all(s is None for s in target_scalers):
            target_scalers = None

        ensemble = cls(
            n_models=len(models),
            layer_sizes=layer_sizes,
            feature_names=m0.feature_names,
            feature_scalers=feature_scalers,
            target_scalers=target_scalers,
        )
        for src, dst in zip(models, ensemble.models):
            dst.load_state_dict(src.state_dict())
        return ensemble

    @classmethod
    def from_folds(cls, run_dir: Path, device: str = "cpu", return_config: bool = False):
        """Build an ensemble by loading one :class:`MuScaRi` model per ``fold_*.pth`` checkpoint."""
        import torch
        ckpt_paths = sorted(Path(run_dir).glob("fold_*.pth"))
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
            model = MuScaRi(
                config.layer_sizes,
                feature_names=feature_names,
                feature_scaler=checkpoint["feature_scaler"],
                target_scaler=checkpoint["target_scaler"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device).eval()
            models.append(model)

        ensemble = cls.from_models(models).eval()
        if return_config:
            return ensemble, config_ref
        return ensemble

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict_mean_sr(self, df: pd.DataFrame) -> np.ndarray:
        """Mean species-richness prediction across ensemble members."""
        return np.mean([m.predict_sr(df) for m in self.models], axis=0).squeeze()

    def get_std_sr(self, df: pd.DataFrame) -> np.ndarray:
        """Standard deviation of species-richness predictions."""
        return np.std([m.predict_sr(df) for m in self.models], axis=0).squeeze()

    def predict_mean_sr_tot(self, df: pd.DataFrame) -> np.ndarray:
        """Mean asymptotic species-richness prediction across ensemble members."""
        return np.mean([m.predict_sr_tot(df) for m in self.models], axis=0).squeeze()

    def get_std_sr_tot(self, df: pd.DataFrame) -> np.ndarray:
        """Standard deviation of asymptotic species-richness predictions."""
        return np.std([m.predict_sr_tot(df) for m in self.models], axis=0).squeeze()


# Backward-compatibility alias
MuScaRiEnsembleModel = MuScaRiEnsemble


