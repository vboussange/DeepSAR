from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin

from muscari.muscari import MuScaRi
from muscari.scaler_serialization import dict_to_scaler, scaler_to_dict


def _normalize_weights(weights: Optional[list], n_models: int) -> list[float]:
    if weights is None:
        return [1.0 / n_models] * n_models
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_models,):
        raise ValueError("ensemble_weights must have one value per model")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(
            "ensemble_weights must be finite, non-negative, and sum to a positive value"
        )
    weights = weights / weights.sum()
    return weights.tolist()


def _validation_rmse(metrics: Optional[dict]) -> float:
    if not metrics:
        return float("nan")
    try:
        if "val_rmse" in metrics:
            return float(metrics["val_rmse"])
        val_metrics = metrics.get("val")
        if isinstance(val_metrics, dict) and "rmse" in val_metrics:
            return float(val_metrics["rmse"])
    except (TypeError, ValueError):
        return float("nan")
    return float("nan")


def _weights_from_member_metrics(member_metrics: list[Optional[dict]]) -> list[float]:
    rmse = np.array([_validation_rmse(metrics) for metrics in member_metrics], dtype=float)
    valid = np.isfinite(rmse) & (rmse > 0)
    if not valid.any():
        return [1.0 / len(member_metrics)] * len(member_metrics)
    fill_value = float(np.median(rmse[valid]))
    rmse = np.where(valid, rmse, fill_value)
    precision = 1.0 / np.maximum(rmse, 1e-12) ** 2
    return (precision / precision.sum()).tolist()


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

    By default, predictions use a uniform ensemble mean. Pass
    ``use_validation_weights=True`` to :meth:`from_folds` or
    :meth:`from_models` to weight members by inverse validation RMSE squared.
    """

    def __init__(
        self,
        n_models: int,
        layer_sizes: list,
        feature_names: list,
        feature_scalers: Optional[list] = None,
        target_scalers: Optional[list] = None,
        muscari_batchnorm: bool = False,
        asymptote_transform: str = "softplus",
        weibull_parameterization: str = "legacy",
        ensemble_weights: Optional[list] = None,
    ):
        super().__init__()
        if n_models <= 0:
            raise ValueError("n_models must be positive")
        for name, scalers in (
            ("feature_scalers", feature_scalers),
            ("target_scalers", target_scalers),
        ):
            if scalers is not None and len(scalers) != n_models:
                raise ValueError(f"{name} must have one entry per model")
        self.n_models = n_models
        self.layer_sizes = layer_sizes
        self.feature_names = feature_names
        self.muscari_batchnorm = muscari_batchnorm
        self.asymptote_transform = asymptote_transform
        self.weibull_parameterization = weibull_parameterization
        self.ensemble_weights = _normalize_weights(ensemble_weights, n_models)
        # Stored as list-of-dicts so config.json stays JSON-serializable
        self.feature_scalers = feature_scalers
        self.target_scalers = target_scalers

        # Convert dicts → sklearn scalers for the sub-models
        _fscalers = (
            [dict_to_scaler(d) if d is not None else None for d in feature_scalers]
            if feature_scalers else [None] * n_models
        )
        _tscalers = (
            [dict_to_scaler(d) if d is not None else None for d in target_scalers]
            if target_scalers else [None] * n_models
        )
        self.models = nn.ModuleList([
            MuScaRi(
                layer_sizes,
                feature_names,
                _fscalers[i],
                _tscalers[i],
                ffnn_batchnorm=self.muscari_batchnorm,
                asymptote_transform=self.asymptote_transform,
                weibull_parameterization=self.weibull_parameterization,
            )
            for i in range(n_models)
        ])

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_models(
        cls,
        models: list[MuScaRi],
        member_metrics: Optional[list] = None,
        ensemble_weights: Optional[list] = None,
        use_validation_weights: bool = False,
    ) -> "MuScaRiEnsemble":
        """Build an ensemble from a list of already-trained :class:`MuScaRi` models."""
        if not models:
            raise ValueError("models list must not be empty")
        m0 = models[0]
        muscari_batchnorm = getattr(m0, "ffnn_batchnorm", False)
        asymptote_transform = getattr(m0, "asymptote_transform", "softplus")
        weibull_parameterization = getattr(m0, "weibull_parameterization", "legacy")

        # Infer layer_sizes from the first model's architecture
        layer_sizes = [block.linear.out_features for block in m0.ffnn.fully_connected_layers]

        for model in models[1:]:
            model_layer_sizes = [
                block.linear.out_features for block in model.ffnn.fully_connected_layers
            ]
            if model.feature_names != m0.feature_names:
                raise ValueError("Feature names differ across ensemble members")
            if model_layer_sizes != layer_sizes:
                raise ValueError("Layer sizes differ across ensemble members")
            for attribute, expected in (
                ("ffnn_batchnorm", muscari_batchnorm),
                ("asymptote_transform", asymptote_transform),
                ("weibull_parameterization", weibull_parameterization),
            ):
                if getattr(model, attribute, None) != expected:
                    raise ValueError(f"{attribute} differs across ensemble members")

        def _to_dict(scaler):
            return scaler_to_dict(scaler) if scaler is not None else None

        feature_scalers = [_to_dict(m.feature_scaler) for m in models]
        target_scalers  = [_to_dict(m.target_scaler)  for m in models]

        # Use None when all scalers are absent
        if all(s is None for s in feature_scalers):
            feature_scalers = None
        if all(s is None for s in target_scalers):
            target_scalers = None
        if member_metrics is not None and len(member_metrics) != len(models):
            raise ValueError("member_metrics must have one entry per model")
        if ensemble_weights is None and use_validation_weights and member_metrics is not None:
            ensemble_weights = _weights_from_member_metrics(member_metrics)

        ensemble = cls(
            n_models=len(models),
            layer_sizes=layer_sizes,
            feature_names=m0.feature_names,
            feature_scalers=feature_scalers,
            target_scalers=target_scalers,
            muscari_batchnorm=muscari_batchnorm,
            asymptote_transform=asymptote_transform,
            weibull_parameterization=weibull_parameterization,
            ensemble_weights=ensemble_weights,
        )
        for src, dst in zip(models, ensemble.models):
            dst.load_state_dict(src.state_dict())
        return ensemble

    @classmethod
    def from_folds(
        cls,
        run_dir: Path,
        device: str = "cpu",
        return_config: bool = False,
        use_validation_weights: bool = False,
    ):
        """Build an ensemble from ``fold_*.pth`` checkpoints.

        Checkpoints are first deserialized on CPU for portability and lower peak
        GPU memory usage. The assembled ensemble is then moved once to ``device``.
        """
        target_device = torch.device(device)
        ckpt_paths = sorted(Path(run_dir).glob("fold_*.pth"))
        if not ckpt_paths:
            raise FileNotFoundError(f"No fold_*.pth files found in {run_dir}")

        models = []
        member_metrics = []
        feature_names_ref = None
        config_ref = None
        for ckpt_path in ckpt_paths:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            feature_names = checkpoint["feature_names"]
            if feature_names_ref is None:
                feature_names_ref = feature_names
                config_ref = checkpoint.get("config")
            else:
                if feature_names_ref != feature_names:
                    raise ValueError("Feature names differ across folds")
            config = checkpoint["config"]
            model = MuScaRi(
                config.layer_sizes,
                feature_names=feature_names,
                feature_scaler=checkpoint["feature_scaler"],
                target_scaler=checkpoint["target_scaler"],
                ffnn_batchnorm=getattr(config, "muscari_batchnorm", False),
                asymptote_transform=getattr(
                    config,
                    "muscari_asymptote_transform",
                    checkpoint.get("asymptote_transform", "softplus"),
                ),
                weibull_parameterization=getattr(
                    config,
                    "muscari_weibull_parameterization",
                    checkpoint.get("weibull_parameterization", "legacy"),
                ),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)
            member_metrics.append(checkpoint.get("metrics"))

        ensemble = cls.from_models(
            models,
            member_metrics=member_metrics,
            use_validation_weights=use_validation_weights,
        ).to(target_device).eval()
        if return_config:
            return ensemble, config_ref
        return ensemble

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def _member_predictions(self, df: pd.DataFrame, predict_method: str) -> np.ndarray:
        preds = [getattr(m, predict_method)(df).reshape(-1) for m in self.models]
        return np.asarray(preds, dtype=float)

    def _as_prediction_matrix(self, predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim != 2:
            raise ValueError("predictions must have shape (n_models, n_samples)")
        if predictions.shape[0] != self.n_models:
            raise ValueError("predictions must have one row per ensemble model")
        return predictions

    def _weighted_mean(self, predictions: np.ndarray) -> np.ndarray:
        weights = np.asarray(self.ensemble_weights)
        return np.average(predictions, axis=0, weights=weights).squeeze()

    def _weighted_std(self, predictions: np.ndarray) -> np.ndarray:
        weights = np.asarray(self.ensemble_weights)
        mean = np.average(predictions, axis=0, weights=weights)
        variance = np.average((predictions - mean) ** 2, axis=0, weights=weights)
        return np.sqrt(np.maximum(variance, 0.0)).squeeze()

    def aggregate_member_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Aggregate a ``(n_models, n_samples)`` prediction matrix."""
        return self._weighted_mean(self._as_prediction_matrix(predictions))

    def get_member_prediction_dispersion(self, predictions: np.ndarray) -> np.ndarray:
        """Configured ensemble dispersion for a ``(n_models, n_samples)`` matrix."""
        return self._weighted_std(self._as_prediction_matrix(predictions))

    def predict_members_sr(self, df: pd.DataFrame) -> np.ndarray:
        """Finite-effort species-richness predictions for each ensemble member."""
        return self._member_predictions(df, "predict_sr")

    def predict_members_sr_tot(self, df: pd.DataFrame) -> np.ndarray:
        """Asymptotic species-richness predictions for each ensemble member."""
        return self._member_predictions(df, "predict_sr_tot")

    def predict_mean_sr(self, df: pd.DataFrame) -> np.ndarray:
        """Mean finite-effort species-richness prediction across members."""
        return self.aggregate_member_predictions(self.predict_members_sr(df))

    def get_std_sr(self, df: pd.DataFrame) -> np.ndarray:
        """Configured ensemble dispersion for finite-effort richness."""
        return self.get_member_prediction_dispersion(self.predict_members_sr(df))

    def predict_mean_sr_tot(self, df: pd.DataFrame) -> np.ndarray:
        """Mean asymptotic species-richness prediction across members."""
        return self.aggregate_member_predictions(self.predict_members_sr_tot(df))

    def get_std_sr_tot(self, df: pd.DataFrame) -> np.ndarray:
        """Configured ensemble dispersion for asymptotic richness."""
        return self.get_member_prediction_dispersion(self.predict_members_sr_tot(df))
