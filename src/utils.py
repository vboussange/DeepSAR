import matplotlib.pyplot as plt
import pickle
import logging

def save_to_pickle(filepath, **kwargs):
    objects_dict = kwargs
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as file:
        pickle.dump(objects_dict, file)
    logging.info(f"Results saved at {filepath}")