import matplotlib.pyplot as plt
import pickle
import logging
import torch

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