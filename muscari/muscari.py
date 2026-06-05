import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from muscari.ffnn import FFNN
from muscari.scaler_serialization import decode_scaler


class MuScaRi(nn.Module):
    """
    Deep SAR model based on the 4-parameter Weibull function.
    """
    _PARAMETER_EPS = 1e-6

    def __init__(
        self,
        layer_sizes: list,
        feature_names: list,
        feature_scaler=None,
        target_scaler=None,
        ffnn_batchnorm: bool = False,
        asymptote_transform: str = "softplus",
    ):
        super().__init__()
        if asymptote_transform not in {"softplus", "exp"}:
            raise ValueError(
                "asymptote_transform must be one of {'softplus', 'exp'}"
            )
        self.layer_sizes = list(layer_sizes)
        self.feature_names = list(feature_names)
        self.feature_scaler = decode_scaler(feature_scaler)
        self.target_scaler = decode_scaler(target_scaler)
        self.ffnn_batchnorm = ffnn_batchnorm
        self.asymptote_transform = asymptote_transform
        self.ffnn = FFNN(
            input_dim=len(self.feature_names),
            layer_sizes=self.layer_sizes,
            output_dim=4,
            batchnorm=ffnn_batchnorm,
        )

    def _weibull_4p(self, x, b, c, d, log_e):
        """4-parameter Weibull: f(x) = c + (d - c) * exp(-exp(b * (ln(x) - log_e)))"""
        log_x = torch.log(torch.clamp(x, min=1e-8))
        inner_exp = torch.clamp(b * (log_x - log_e), min=-50, max=50)
        outer_exp = torch.clamp(-torch.exp(inner_exp), min=-50, max=0)
        return c + (d - c) * torch.exp(outer_exp)

    def _predict_b_c_d_e(self, x):
        x = self.ffnn(x)
        b = F.softplus(x[:, 0:1]) + self._PARAMETER_EPS
        raw_c = x[:, 1:2]
        if self.asymptote_transform == "exp":
            c = torch.exp(torch.clamp(raw_c, min=-20, max=20)) + self._PARAMETER_EPS
        else:
            c = F.softplus(raw_c) + self._PARAMETER_EPS
        d = c * torch.sigmoid(x[:, 2:3])  # ensure 0 < d < c
        log_e = x[:, 3:4]
        return b, c, d, log_e

    def forward(self, x):
        log_aplot, features = x[:, :1], x[:, 1:]
        b, c, d, log_e = self._predict_b_c_d_e(features)
        return self._weibull_4p(log_aplot, b, c, d, log_e)

    def _predict_sr_tot(self, x):
        """Predict asymptotic SR; ``log_aplot`` must not be in ``x``."""
        _, c, _, _ = self._predict_b_c_d_e(x)
        return c

    def predict_sr(self, df: pd.DataFrame):
        """Predict species richness given sampling effort (column ``log_observed_area``)."""
        self.eval()
        with torch.no_grad():
            X = df[["log_observed_area"] + self.feature_names].values.astype(np.float32)
            if self.feature_scaler is not None:
                X = self.feature_scaler.transform(X)
            X = torch.tensor(X, dtype=torch.float32).to(next(self.parameters()).device)
            y_pred = self(X).cpu().numpy()
            if self.target_scaler is not None:
                y_pred = self.target_scaler.inverse_transform(y_pred)
            return y_pred

    def predict_sr_tot(self, df: pd.DataFrame):
        """Predict total (asymptotic) species richness."""
        self.eval()
        with torch.no_grad():
            x = df[self.feature_names].values.astype(np.float32)
            x = np.concatenate([np.zeros((x.shape[0], 1)), x], axis=1)
            if self.feature_scaler is not None:
                x = self.feature_scaler.transform(x)
            x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)
            x = x[:, 1:]  # drop the dummy log_observed_area column
            y_pred = self._predict_sr_tot(x)
            if self.target_scaler is not None:
                y_pred = self.target_scaler.inverse_transform(y_pred.cpu().numpy())
            return y_pred

    @staticmethod
    def initialize(checkpoint, device="cuda"):
        """Load a single model from a saved checkpoint."""
        config = checkpoint["config"]
        model = MuScaRi(
            config.layer_sizes,
            feature_names=checkpoint["feature_names"],
            feature_scaler=checkpoint["feature_scaler"],
            target_scaler=checkpoint["target_scaler"],
            ffnn_batchnorm=getattr(config, "muscari_batchnorm", False),
            asymptote_transform=getattr(
                config,
                "muscari_asymptote_transform",
                checkpoint.get("asymptote_transform", "softplus"),
            ),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device).eval()
