from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler


SUPPORTED_SCALER_TYPES = (MinMaxScaler, MaxAbsScaler, StandardScaler)


def scaler_to_dict(scaler) -> dict:
    """Serialize a fitted sklearn scaler to a JSON-compatible dict."""
    cls_name = type(scaler).__name__
    data = {"cls": cls_name, "n_features_in_": int(scaler.n_features_in_)}
    if cls_name == "MinMaxScaler":
        data["min_"] = scaler.min_.tolist()
        data["scale_"] = scaler.scale_.tolist()
        data["data_min_"] = scaler.data_min_.tolist()
        data["data_max_"] = scaler.data_max_.tolist()
        data["data_range_"] = scaler.data_range_.tolist()
    elif cls_name == "MaxAbsScaler":
        data["scale_"] = scaler.scale_.tolist()
        data["max_abs_"] = scaler.max_abs_.tolist()
    elif cls_name == "StandardScaler":
        data["mean_"] = scaler.mean_.tolist()
        data["scale_"] = scaler.scale_.tolist()
    else:
        raise TypeError(f"Unsupported scaler type: {cls_name}")
    return data


def dict_to_scaler(data: dict):
    """Reconstruct a scaler from a serialized dict."""
    cls_name = data["cls"]
    if cls_name == "MinMaxScaler":
        scaler = MinMaxScaler()
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
        scaler = StandardScaler()
        scaler.mean_ = np.array(data["mean_"])
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
}