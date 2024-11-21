import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the basic neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
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

# Define the basic neural network model
class SimpleNNBatchNorm(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(SimpleNNBatchNorm, self).__init__()
        layer_sizes = [input_dim, 16, 16, 16]
        self.fully_connected_layers = nn.ModuleList(
            [FullyConnectedBatchNormBlock(in_f, out_f) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.last_fully_connected = nn.Linear(layer_sizes[-1], output_dim)

    def forward(self, x):
        for fully_connected_layer in self.fully_connected_layers:
            x = fully_connected_layer(x)
        x = self.last_fully_connected(x)
        return x
    
# Species Area Environment model combining area and environment features
class NNSAR(nn.Module):
    def __init__(self, input_dim):
        super(NNSAR, self).__init__()
        self.nn1 = SimpleNN(input_dim, 1)
        self.nn2 = SimpleNN(input_dim, 1)

    def forward(self, env_pred, log_area):
        return self.nn1(env_pred) + log_area * self.nn2(env_pred)

# SAR where c and z are outputs of the same nn, theophile version with batchnorm
class NNSAR2(nn.Module):
    def __init__(self, input_dim):
        super(NNSAR2, self).__init__()
        self.nn = SimpleNNBatchNorm(input_dim, 2)


    def forward(self, env_pred, log_area):
        x = self.nn(env_pred)
        return  x[:,0].reshape(-1, 1) + log_area * F.relu(x[:,1].reshape(-1, 1))

# SAR where c and z are outputs of the same nn
class NNSAR3(nn.Module):
    def __init__(self, input_dim):
        super(NNSAR3, self).__init__()
        self.nn = SimpleNN(input_dim, 2)

    def forward(self, env_pred, log_area):
        x = self.nn(env_pred)
        return  x[:,0].reshape(-1, 1) + log_area * x[:,1].reshape(-1, 1)
    
     
# SAR where c and z are outputs of the same nn with different inputs, theophile version with batchnorm
class NNSAR4(nn.Module):
    def __init__(self, input_dim_c, input_dim_z):
        super(NNSAR4, self).__init__()
        self.c = SimpleNNBatchNorm(input_dim_c, 1)
        self.z = SimpleNNBatchNorm(input_dim_z, 1)


    def forward(self, mean_env, std_env, log_area):
        c = self.c(mean_env)
        z = F.relu(self.z(std_env))
        return  c.reshape(-1, 1) + log_area * z.reshape(-1, 1)
    
# SAR
class SAR(nn.Module):
    def __init__(self):
        super(SAR, self).__init__()
        self.c = nn.parameter.Parameter(data=torch.rand(1) - 0.5)
        self.z = nn.parameter.Parameter(data=torch.rand(1))

    def forward(self, log_area):
        return  self.c + F.relu(self.z) * log_area