import matplotlib.pyplot as plt
import pickle
import logging
import torch
import torch.nn as nn

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
        
        
def save_to_pickle(filepath, **kwargs):
    objects_dict = kwargs
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as file:
        pickle.dump(objects_dict, file)
    logging.info(f"Results saved at {filepath}")

def symmetric_arch(n, base=32, factor=2):
    half = (n + 1) // 2
    front = [base * factor**i for i in range(half)]
    mirror = front[:-1] if n % 2 else front
    return front + mirror[::-1]