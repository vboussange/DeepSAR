import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# TODO: work in progress


class FullyConnectedBatchNormBlock(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(FullyConnectedBatchNormBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x)
        return x
    
class Neural4PWeibull(nn.Module):
    def __init__(self, n_features, layer_sizes, p0):
        super(Neural4PWeibull, self).__init__()
        layer_sizes = [n_features] + layer_sizes
        
        self.fully_connected_layers = nn.ModuleList(
            [FullyConnectedBatchNormBlock(in_f, out_f) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])])
        
        # last layer corresponds to the parameters of the Weibull function
        self.last_fully_connected = nn.Linear(layer_sizes[-1], 4)
        # p0 corresponds to [b, c, d, e]; need to transform it to [b, c, d_offset, e]
        p0[2] = p0[2] - p0[1]  # d_offset = d - c
        assert p0[3] > 0, "e must be positive"
        
        self.last_fully_connected.bias.data = torch.tensor(p0, dtype=torch.float32)
        nn.init.normal_(self.last_fully_connected.weight, mean=0, std=0.01)
        
    def weibull_4p(self, x, b, c, d, e):
        """
        4-parameter Weibull function: f(x) = c + (d - c) * exp(-exp(b * (ln(x) - ln(e))))
        """
    
        # More stable computation
        log_x = torch.log(torch.clamp(x, min=1e-8))
        log_e = torch.log(torch.clamp(e, min=1e-8))
        return c + (d - c) * torch.exp(-torch.exp(b * (log_x - log_e)))


    def predict_b_c_d_e(self, x):
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
        b, c, d, e = self.predict_b_c_d_e(features)
        sr = F.softplus(self.weibull_4p(log_aplot, b, c, d, e))
        # log_sr = torch.log(sr)
        return sr
    
    def predict_sr(self, x):
        """
        Predicts asymptotic SR from features. `log_aplot` should not appear in `x`.
        """
        b, c, d, e = self.predict_b_c_d_e(x)
        sr = F.softplus(e)
        return sr
    
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

if __name__ == "__main__":
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # 4-parameter Weibull function

    # Generate synthetic data
    n_features = 5
    layer_sizes = [16, 8]
    # layer_sizes = []
    p0 = [1., 6.0, -1, 3.0]  # [b, c, d_offset, e_offset]

    batch_size = 128
    n_samples = 1000

    # True Weibull parameters
    b_true = 1.
    c_true = 8.0
    d_true = -1.
    e_true = 3. # inflection point in log space

    # Generate random features and aplot
    np.random.seed(42)
    torch.manual_seed(42)
    features = np.random.randn(n_samples, n_features)
    log_aplot = np.random.rand(n_samples, 1) * 10 + 1  # avoid log(0)

    model = Neural4PWeibull(n_features=n_features, 
                            layer_sizes=layer_sizes, 
                            p0=p0)

    # Generate targets using the 4-parameter Weibull function
    y = model.weibull_4p(torch.tensor(log_aplot, dtype=torch.float32), 
                         torch.tensor(b_true, dtype=torch.float32), 
                         torch.tensor(c_true, dtype=torch.float32), 
                         torch.tensor(d_true, dtype=torch.float32), 
                         torch.tensor(e_true, dtype=torch.float32))
    y = y.squeeze() + 0.2 * np.random.normal(0, 0.1, size=(n_samples,))

    # Plot the generated targets
    plt.figure(figsize=(10, 6))
    plt.plot(log_aplot.squeeze(), y, 'b.', alpha=0.6, label='Generated targets (with noise)')
    plt.xlabel('aplot')
    plt.ylabel('y (Weibull output)')
    plt.title('Generated Weibull Data with Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Prepare input tensor
    x = np.concatenate([log_aplot, features], axis=1)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Model, loss, optimizer
    criterion = MSELogLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    n_epochs = 400
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

    # Evaluate the model to get predicted parameters
    model.eval()
    with torch.no_grad():
        # Use the same input to get predicted parameters
        b_pred, c_pred, d_pred, e_pred = model.predict_b_c_d_e(x_tensor[:, 1:])
        
        # Calculate mean predicted parameters
        b_mean = b_pred.mean().item()
        c_mean = c_pred.mean().item()
        d_mean = d_pred.mean().item()
        e_mean = e_pred.mean().item()
        
    
    # Generate predictions for plotting
    y_pred = model(x_tensor).squeeze()

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
    print("\n" + "="*50)
    print("PARAMETER COMPARISON")
    print("="*50)
    print(f"{'Parameter':<12} {'True Value':<12} {'Predicted':<12} {'Error':<12}")
    print("-"*50)
    print(f"{'b':<12} {b_true:<12.4f} {b_mean:<12.4f} {abs(b_true - b_mean):<12.4f}")
    print(f"{'c':<12} {c_true:<12.4f} {c_mean:<12.4f} {abs(c_true - c_mean):<12.4f}")
    print(f"{'d':<12} {d_true:<12.4f} {d_mean:<12.4f} {abs(d_true - d_mean):<12.4f}")
    print(f"{'e':<12} {e_true:<12.4f} {e_mean:<12.4f} {abs(e_true - e_mean):<12.4f}")
    print("="*50)