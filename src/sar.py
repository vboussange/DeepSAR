import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SAR(nn.Module):
    def __init__(self):
        super(SAR, self).__init__()
        self.c = nn.parameter.Parameter(data=torch.rand(1) - 0.5)
        self.z = nn.parameter.Parameter(data=torch.rand(1))

    def forward(self, log_area):
        return  self.c + F.relu(self.z) * log_area