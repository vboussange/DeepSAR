from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler


class Log1pMaxScaler:
    """Scale positive richness values by max log1p richness."""

    def fit(self, x, y=None):
        x = np.asarray(x, dtype=float)
        if np.any(x < 0):
            raise ValueError("Log1pMaxScaler expects non-negative values")
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.n_features_in_ = x.shape[1]
        self.scale_ = np.maximum(np.nanmax(np.log1p(x), axis=0), 1e-12)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return np.log1p(np.maximum(x, 0.0)) / self.scale_

    def fit_transform(self, x, y=None):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return np.expm1(x * self.scale_)


SUPPORTED_SCALER_TYPES = (MinMaxScaler, MaxAbsScaler, StandardScaler, Log1pMaxScaler)


def scaler_to_dict(scaler) -> dict:
    """Serialize a fitted sklearn scaler to a JSON-compatible dict."""
    if not isinstance(scaler, SUPPORTED_SCALER_TYPES):
        raise TypeError(f"Unsupported scaler type: {type(scaler).__name__}")
    cls_name = type(scaler).__name__
    data = {"cls": cls_name, "n_features_in_": int(scaler.n_features_in_)}
    if cls_name == "MinMaxScaler":
        data["feature_range"] = list(scaler.feature_range)
        data["clip"] = scaler.clip
        data["min_"] = scaler.min_.tolist()
        data["scale_"] = scaler.scale_.tolist()
        data["data_min_"] = scaler.data_min_.tolist()
        data["data_max_"] = scaler.data_max_.tolist()
        data["data_range_"] = scaler.data_range_.tolist()
    elif cls_name == "MaxAbsScaler":
        data["scale_"] = scaler.scale_.tolist()
        data["max_abs_"] = scaler.max_abs_.tolist()
    elif cls_name == "StandardScaler":
        data["with_mean"] = scaler.with_mean
        data["with_std"] = scaler.with_std
        data["mean_"] = scaler.mean_.tolist() if scaler.mean_ is not None else None
        data["scale_"] = scaler.scale_.tolist() if scaler.scale_ is not None else None
    elif cls_name == "Log1pMaxScaler":
        data["scale_"] = scaler.scale_.tolist()
    else:
        raise TypeError(f"Unsupported scaler type: {cls_name}")
    return data


def dict_to_scaler(data: dict):
    """Reconstruct a scaler from a serialized dict."""
    cls_name = data["cls"]
    if cls_name == "MinMaxScaler":
        scaler = MinMaxScaler(
            feature_range=tuple(data.get("feature_range", (0, 1))),
            clip=data.get("clip", False),
        )
        scaler.min_ = np.array(data["min_"])
        scaler.scale_ = np.array(data["scale_"])
        scaler.data_min_ = np.array(data["data_min_"])
        scaler.data_max_ = np.array(data["data_max_"])
        scaler.data_range_ = np.array(data["data_range_"])
    elif cls_name == "MaxAbsScaler":
        scaler = MaxAbsScaler()
        scaler.scale_ = np.array(data["scale_"])
        scaler.max_abs_ = np.array(data["max_abs_"])
    elif cls_name == "StandardScaler":
        scaler = StandardScaler(
            with_mean=data.get("with_mean", True),
            with_std=data.get("with_std", True),
        )
        scaler.mean_ = np.array(data["mean_"]) if data["mean_"] is not None else None
        scaler.scale_ = np.array(data["scale_"]) if data["scale_"] is not None else None
    elif cls_name == "Log1pMaxScaler":
        scaler = Log1pMaxScaler()
        scaler.scale_ = np.array(data["scale_"])
    else:
        raise TypeError(f"Unsupported scaler type: {cls_name}")
    scaler.n_features_in_ = data["n_features_in_"]
    return scaler


def decode_scaler(scaler):
    if scaler is None or isinstance(scaler, SUPPORTED_SCALER_TYPES):
        return scaler
    if isinstance(scaler, dict):
        return dict_to_scaler(scaler)
    raise TypeError(f"Unsupported scaler payload type: {type(scaler).__name__}")


SCALER_CODERS = {
    MinMaxScaler: (scaler_to_dict, dict_to_scaler),
    MaxAbsScaler: (scaler_to_dict, dict_to_scaler),
    StandardScaler: (scaler_to_dict, dict_to_scaler),
    Log1pMaxScaler: (scaler_to_dict, dict_to_scaler),
}
