import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from deepsar.ensemble_model import DeepSAREnsembleModel
from deepsar.deepsar_model import DeepSARModel

class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(FullyConnectedBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features, **kwargs)
    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(x)
        return x
    
class Deep4PWeibull(DeepSARModel):
    """
    Deep SAR model based on the 4-parameter Weibull function.
    """
    def __init__(self,
                 layer_sizes, 
                 feature_names,
                 feature_scaler=None, 
                 target_scaler=None):
        super(Deep4PWeibull, self).__init__(feature_names=feature_names, 
                                            feature_scaler=feature_scaler, 
                                            target_scaler=target_scaler)
        layer_sizes = [len(feature_names)] + layer_sizes
        
        self.fully_connected_layers = nn.ModuleList(
            [FullyConnectedBlock(in_f, out_f) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.last_fully_connected = nn.Linear(layer_sizes[-1], 4)

        # p0[2] = p0[1] - p0[2]   # d_offset = c - d
        # self.last_fully_connected.bias.data = torch.tensor(p0, dtype=torch.float32)
        # nn.init.xavier_normal_(self.last_fully_connected.weight, gain=0.05)
        
    def _weibull_4p(self, x, b, c, d, e):
        """
        4-parameter Weibull function: f(x) = c + (d - c) * exp(-exp(b * (ln(x) - ln(e))))
        """
        log_x = torch.log(torch.clamp(x, min=1e-8))
        log_e = torch.log(torch.clamp(e, min=1e-8))
        # Clamp the inner exponential to prevent overflow
        inner_exp = torch.clamp(b * (log_x - log_e), min=-50, max=50)
        outer_exp = torch.clamp(-torch.exp(inner_exp), min=-50, max=0)
        return c + (d - c) * torch.exp(outer_exp)


    def _predict_b_c_d_e(self, x):
        for fully_connected_layer in self.fully_connected_layers:
            x = fully_connected_layer(x)
        x = self.last_fully_connected(x)
        
        # Extract and constrain parameters
        b = x[:, 0:1]
        c = x[:, 1:2]
        d = c - F.softplus(x[:, 2:3])  # Ensure d < c
        e = F.softplus(x[:, 3:4])      # Ensure e > 0
        return b, c, d, e

    def forward(self, x):
        log_aplot, features = x[:, :1], x[:, 1:]
        b, c, d, e = self._predict_b_c_d_e(features)
        sr = self._weibull_4p(log_aplot, b, c, d, e)
        return sr
    
    def _predict_sr_tot(self, x):
        """
        Predicts asymptotic SR from features. `log_aplot` should not appear in `x`.
        """
        _, c, _, _ = self._predict_b_c_d_e(x)
        return c
    
    @staticmethod
    def initialize(checkpoint, device="cuda"):
        """Load the model and scalers from the saved checkpoint."""
        feature_names = checkpoint["feature_names"]
        model_state = checkpoint["model_state_dict"]
        config = checkpoint["config"]
        feature_scaler = checkpoint["feature_scaler"]
        target_scaler = checkpoint["target_scaler"]
        model = Deep4PWeibull(config.layer_sizes,
                              feature_names=feature_names,
                              feature_scaler=feature_scaler,
                              target_scaler=target_scaler)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        return model
    
    @staticmethod
    def initialize_ensemble(checkpoint, device="cuda"):
        """Load the model and scalers from the saved checkpoint."""
        feature_names = checkpoint["feature_names"]
        model_state = checkpoint["ensemble_model_state_dict"]
        config = checkpoint["config"]
        feature_scalers = checkpoint["feature_scalers"]
        target_scalers = checkpoint["target_scalers"]
        models = []
        for i in range(config.n_ensembles):
            models.append(Deep4PWeibull(config.layer_sizes,
                                        feature_names=feature_names,
                                        feature_scaler=feature_scalers[i],
                                        target_scaler=target_scalers[i]))
        ensemble_model = DeepSAREnsembleModel(models)
        ensemble_model.load_state_dict(model_state)
        ensemble_model = ensemble_model.to(device)
        ensemble_model.eval()
        return ensemble_model


if __name__ == "__main__":
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # 4-parameter Weibull function

    # Generate synthetic data
    n_features = 5
    layer_sizes = [16, 8]

    batch_size = 256
    n_samples = 1000

    # True Weibull parameters
    b_true = 1e-1
    c_true = 1000
    d_true = 10
    e_true = 500 # inflection point in log space

    # Generate random features and aplot
    np.random.seed(42)
    torch.manual_seed(42)
    features = np.random.randn(n_samples, n_features)
    log_aplot = np.random.rand(n_samples, 1) * 10 + 1  # avoid log(0)

    model = Deep4PWeibull(feature_names=range(n_features), layer_sizes=layer_sizes)

    # Generate targets using the 4-parameter Weibull function
    y = model._weibull_4p(torch.tensor(log_aplot, dtype=torch.float32), 
                         torch.tensor(b_true, dtype=torch.float32), 
                         torch.tensor(c_true, dtype=torch.float32), 
                         torch.tensor(d_true, dtype=torch.float32), 
                         torch.tensor(e_true, dtype=torch.float32))
    y = y.squeeze() + 0.5 * np.random.normal(0, 0.1, size=(n_samples,))
    
    # Min Max scaling
    y_scaled = (y - y.min()) / (y.max() - y.min())  # Scale to [0, 1]

    # Prepare input tensor
    x = np.concatenate([log_aplot, features], axis=1)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # Model, loss, optimizer
    criterion = nn.MSELoss()
    # criterion = MSELogLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.1)

    # Training loop
    n_epochs = 20
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        x_shuffled = x_tensor[perm]
        y_shuffled = y_tensor[perm]

        for i in range(0, n_samples, batch_size):
            xb = x_shuffled[i:i+batch_size]
            y_b = y_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred.squeeze(), y_b)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    # # Evaluate the model to get predicted parameters
    # model.eval()
    # with torch.no_grad():
    #     # Use the same input to get predicted parameters
    #     b_pred, c_pred, d_pred, e_pred = model._predict_b_c_d_e(x_tensor[:, 1:])
        
    #     # Calculate mean predicted parameters
    #     b_mean = b_pred.mean().item()
    #     c_mean = c_pred.mean().item()
    #     d_mean = d_pred.mean().item()
    #     e_mean = e_pred.mean().item()
        
    
    # Generate predictions for plotting
    y_pred = model(x_tensor).squeeze()* (y.max() - y.min()) + y.min()

    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.scatter(log_aplot.squeeze(), y.squeeze(), alpha=0.6, label='True values', color='blue', s=20)
    plt.scatter(log_aplot.squeeze(), y_pred.detach().numpy(), alpha=0.6, label='Predictions', color='red', s=20)
    plt.xlabel('aplot')
    plt.ylabel('y')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Pretty print comparison
    # print("\n" + "="*50)
    # print("PARAMETER COMPARISON")
    # print("="*50)
    # print(f"{'Parameter':<12} {'True Value':<12} {'Predicted':<12} {'Error':<12}")
    # print("-"*50)
    # print(f"{'b':<12} {b_true:<12.4f} {b_mean:<12.4f} {abs(b_true - b_mean):<12.4f}")
    # print(f"{'c':<12} {c_true:<12.4f} {c_mean:<12.4f} {abs(c_true - c_mean):<12.4f}")
    # print(f"{'d':<12} {d_true:<12.4f} {d_mean:<12.4f} {abs(d_true - d_mean):<12.4f}")
    # print(f"{'e':<12} {e_true:<12.4f} {e_mean:<12.4f} {abs(e_true - e_mean):<12.4f}")
    # print("="*50)