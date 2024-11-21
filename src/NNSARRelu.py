import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.NNSAR import SimpleNNBatchNorm
# SAR where c and z are outputs of the same nn, theophile version with batchnorm
class NNSARRelu(nn.Module):
    def __init__(self, input_dim):
        super(NNSARRelu, self).__init__()
        self.nn = SimpleNNBatchNorm(input_dim, 3)


    def forward(self, env_pred, log_area):
        x = self.nn(env_pred)
        logc = x[:,0].reshape(-1, 1)
        z = x[:,1].reshape(-1, 1)
        cutoff = x[:,2].reshape(-1, 1)
        return  logc + torch.relu(log_area - cutoff) * z
