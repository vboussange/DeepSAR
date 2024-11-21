import matplotlib.pyplot as plt
import pickle
import logging

def save_to_pickle(filepath, **kwargs):
    objects_dict = kwargs
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as file:
        pickle.dump(objects_dict, file)
    logging.info(f"Results saved at {filepath}")


HABITATS_ALL = ["forest_t1", "forest_t3", "grass_r3", "grass_r4", "scrub_s2", "scrub_s6"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"][0:len(HABITATS_ALL)];