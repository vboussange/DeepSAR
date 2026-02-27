import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=False, **kwargs):
        super(FullyConnectedBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.batch_norm = nn.BatchNorm1d(out_features) if batchnorm else None

    def forward(self, x):
        x = self.linear(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = F.leaky_relu(x)
        return x
    
class FFNN(nn.Module):
    def __init__(self, input_dim, layer_sizes, output_dim=1, batchnorm=False):
        super(FFNN, self).__init__()
        layer_sizes = [input_dim] + layer_sizes
        self.fully_connected_layers = nn.ModuleList(
            [
                FullyConnectedBlock(in_f, out_f, batchnorm=batchnorm)
                for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])
            ]
        )
        self.last_fully_connected = nn.Linear(layer_sizes[-1], output_dim)

    def forward(self, x):
        for fully_connected_layer in self.fully_connected_layers:
            x = fully_connected_layer(x)
        x = self.last_fully_connected(x)
        return x

class FFNNExp(nn.Module):
    def __init__(self, input_dim, layer_sizes, batchnorm=False):
        super(FFNNExp, self).__init__()
        self.nn = FFNN(input_dim, layer_sizes, 1, batchnorm=batchnorm)

    def forward(self, preds):
        x = self.nn(preds)
        return torch.exp(x)

def load_model_checkpoint(model_state, predictors, layer_sizes, batchnorm=False):
        """Load the model and scalers from the saved checkpoint."""
        model = FFNNExp(len(predictors), layer_sizes=layer_sizes, batchnorm=batchnorm)
        model.load_state_dict(model_state)
        model.eval()
        return model