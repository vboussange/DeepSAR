import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
# https://github.com/jlager/BINNs/blob/master/Modules/Utils/Gradient.py


class FullyConnectedBatchNormBlock(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(FullyConnectedBatchNormBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x
    
    
class SimpleNNBatchNorm(nn.Module):
    def __init__(self, input_dim, layer_sizes, output_dim=1):
        super(SimpleNNBatchNorm, self).__init__()
        layer_sizes = [input_dim] + layer_sizes
        self.fully_connected_layers = nn.ModuleList(
            [FullyConnectedBatchNormBlock(in_f, out_f) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.last_fully_connected = nn.Linear(layer_sizes[-1], output_dim)

    def forward(self, x):
        for fully_connected_layer in self.fully_connected_layers:
            x = fully_connected_layer(x)
        x = self.last_fully_connected(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(MLP, self).__init__()
        self.nn = SimpleNNBatchNorm(input_dim, layer_sizes, 1)

    def forward(self, preds):
        x = self.nn(preds)
        return x

def get_gradient(outputs, inputs):
    outputs = outputs.sum()
    grads = grad(outputs, inputs, create_graph=True)[0]
    return grads
    
class CustomMSELoss(nn.Module):
    # https://github.com/jlager/BINNs/blob/master/Modules/Models/BuildBINNs.py
    def __init__(self, dSRdA_weight=1e5):
        super(CustomMSELoss, self).__init__()
        self.dSRdA_weight = dSRdA_weight

    def forward(self, model, predictions, input_features, targets):
        loss_mse = torch.mean((predictions - targets) ** 2)
        try:
            # # Gradient penalty on training data
            # grads = torch.autograd.grad(predictions.sum(), input_features, create_graph=True)[0]
            # grads_a = grads[:, 0]
            # loss_grad = self.dSRdA_weight * torch.mean((torch.relu(-grads_a))**2)
            
            # Random sampling across data range
            batch_size = input_features.size(0)
            random_inputs = torch.rand(batch_size, input_features.size(1), device=input_features.device)
            random_inputs.requires_grad_(True)

            # Gradient penalty on random inputs
            random_outputs = model(random_inputs)
            grads_random = torch.autograd.grad(random_outputs.sum(), random_inputs, create_graph=True)[0]
            grads_a_random = grads_random[:, 0]
            loss_grad_random = self.dSRdA_weight * torch.mean(torch.relu(-grads_a_random) ** 2)

        except Exception as e:
            print(f"Problem with gradient evaluation: {e}")
            loss_grad = 0
        return loss_mse + loss_grad_random #+ loss_grad
    
    
def scale_feature_tensor(x, scaler):
    assert len(x.shape)
    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float32, device = x.device).reshape(1, -1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float32, device = x.device).reshape(1, -1)
    features_scaled = (x - mean_tensor) / scale_tensor
    return features_scaled

def inverse_transform_scale_feature_tensor(y, scaler):
    assert len(y.shape)
    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float32, device = y.device).reshape(1, -1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float32, device = y.device).reshape(1, -1)
    invy = y * scale_tensor + mean_tensor
    return invy


# We need to calculate gradient wrt max - min lat long and area, but we can also directly calculate gradient based on area later on to plot results

if __name__ == "__main__":
    def test_model_learns_with_gradient_constraint():
        torch.manual_seed(0)

        input_dim = 5  
        batch_size = 100
        X_test = torch.randn(batch_size, input_dim, requires_grad=True)  # Random input features
        y_test = -0.1 * X_test[:,0] + 10 * X_test[:,1] + 12 * X_test[:,2]
        y_test = y_test.reshape(-1,1).detach()
        # Initialize the linear model
        model = nn.Linear(input_dim, 1)

        loss_fn = CustomMSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

        num_epochs = 500
        for epoch in range(num_epochs):
            model.train()

            # Forward pass
            predictions = model(X_test)
            
            # Compute the loss
            loss = loss_fn(predictions, X_test, y_test)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print the loss for tracking
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

        # testing
        model.eval()
        predictions = model(X_test)
        grads = get_gradient(predictions, X_test)
        grads_a = grads[:, :3]
        assert torch.all(grads_a >= 0)

    test_model_learns_with_gradient_constraint()
    
    def test_skorch():
        CONFIG = {
                    "torch_params": {
                        "optimizer":optim.Adam,
                        "lr":5e-1,
                        "batch_size":256,
                        "max_epochs":500,
                        # "callbacks":[("lr_scheduler", LRScheduler(policy = "ReduceLROnPlateau", monitor='valid_loss', patience=20, factor=0.5))],
                        # "optimizer__weight_decay":1e-4,
                        "train_split": None,
                        "device": torch.device("cpu")
                    },
                }
        input_dim = 5
        X_test = torch.randn(CONFIG["torch_params"]["batch_size"], input_dim, requires_grad=True)  # Random input features
        y_test = -0.1 * X_test[:,0] + 10 * X_test[:,1] + 12 * X_test[:,2]
        y_test = y_test.reshape(-1,1).detach()
        reg = MyNet(module=nn.Linear,
                    module__in_features=input_dim,
                    module__out_features=1,
                    criterion=CustomMSELoss,
                    **CONFIG["torch_params"])
        reg.fit(X_test, y_test)
        

    
    def misc_test():
        input_features = torch.randn(100, 10, requires_grad=True)
        model = MLP(10)
        outputs = model(input_features)
        grads = get_gradient(outputs, input_features)
        grads_a = grads[:, :3]
        torch.where(grads_a < 0, grads_a**2, torch.zeros_like(grads_a))
    
    
