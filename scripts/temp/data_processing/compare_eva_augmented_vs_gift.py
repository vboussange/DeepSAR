import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

path_data = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "EVA_vs_GIFT_compilation"

eva_augmented = gpd.read_file(path_data / "EVA_augmented_data.gpkg")
gift_data = gpd.read_file(path_data / "GIFT_data.gpkg")

gift_data["log_sr"] = np.log(gift_data["sr"])
eva_augmented["log_sr"] = np.log(eva_augmented["sr"].astype(float))

eva_augmented = eva_augmented[np.isfinite(eva_augmented["log_sr"])]
gift_data = gift_data.loc[eva_augmented.index]

fig, ax = plt.subplots()
r2 = r2_score(eva_augmented.log_sr, gift_data.log_sr)
corr = np.corrcoef(eva_augmented.log_sr, gift_data.log_sr)[0, 1]
ax.scatter(eva_augmented.log_sr, gift_data.log_sr)
ax.set_xlabel("log(SR) EVA")
ax.set_ylabel("log(SR) GIFT")

min_val = min(eva_augmented.log_sr.min(), gift_data.log_sr.min())
max_val = max(eva_augmented.log_sr.max(), gift_data.log_sr.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
ax.text(0.05, 0.95, f"R2: {r2:.4f}\nCorr: {corr:.4f}", transform=plt.gca().transAxes,
        verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
ax.legend()
