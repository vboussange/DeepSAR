"""
Predicts SAR from an ensembled DeepSAR model at specified locations.

Using Ensemble model.
"""
import numpy as np
import pandas as pd
from deepsar.utils import save_to_pickle, load_ensemble_from_folds
from deepsar.data_processing.utils_features import EnvironmentalFeatureDataset
import matplotlib.pyplot as plt

from pathlib import Path
from pyproj import Transformer

ROOT = Path(__file__).parents[2]
TRAINING_DATASET_SEED = "ceacce0"
RUN_DIR = ROOT / "scripts" / "results" / "train" / f"{TRAINING_DATASET_SEED}_no_lc_features_reduced_bioclim_vars"

def load_environmental_features(model):
    env_features = EnvironmentalFeatureDataset()
    env_ds, lc_ds = env_features.load(use_cache=True)
    env_ds = env_ds.rio.write_crs("EPSG:3035")
    lc_ds = lc_ds.rio.write_crs("EPSG:3035")

    env_vars = [
        v for v in env_ds.data_vars
        if (v in model.feature_names) or (f"std_{v}" in model.feature_names)
    ]
    env_ds = env_ds[env_vars]
    res_env_pixel = abs(env_ds.rio.resolution()[0])
    return env_ds, lc_ds, res_env_pixel


def build_features_for_window(model, env_ds, lc_ds, x, y, window_size, res_env_pixel):
    if window_size < res_env_pixel:
        reduced_env = env_ds.sel(x=x, y=y, method="nearest")
    else:
        reduced_env = env_ds.sel(
            x=slice(x, x + window_size),
            y=slice(y, y - window_size),
        )

    feature_values = {}
    for var in reduced_env.data_vars:
        feature_values[var] = reduced_env[var].mean().item()
        feature_values[f"std_{var}"] = reduced_env[var].std().item()

    lc_frac_cols = [c for c in model.feature_names if c.startswith("lc_frac_")]
    if lc_frac_cols:
        lc_da = lc_ds["landcover"].where(lc_ds["landcover"] >= 0)
        if window_size < res_env_pixel:
            lc_sel = lc_da.sel(x=x, y=y, method="nearest")
            for col in lc_frac_cols:
                idx = int(col.split("_")[-1])
                feature_values[col] = float(lc_sel == idx)
        else:
            lc_window = lc_da.sel(
                x=slice(x, x + window_size),
                y=slice(y, y - window_size),
            )
            for col in lc_frac_cols:
                idx = int(col.split("_")[-1])
                feature_values[col] = (lc_window == idx).mean().item()

    if "log_sp_unit_area" in model.feature_names:
        feature_values["log_sp_unit_area"] = np.log(window_size**2)

    missing = [name for name in model.feature_names if name not in feature_values]
    if missing:
        raise ValueError(f"Missing features for prediction: {missing}")

    return pd.DataFrame([{name: feature_values[name] for name in model.feature_names}])


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def export_sar_table(locations, window_sizes, output_path: Path) -> None:
    area_km2 = (window_sizes ** 2) / 1e6
    area_targets = np.array([5e3**2 / 1e6, 5e4**2 / 1e6])
    idx_low = nearest_index(area_km2, area_targets[0])
    idx_high = nearest_index(area_km2, area_targets[1])

    rows = []
    label_map = {"loc1": "A", "loc2": "B", "loc3": "C"}
    for name in label_map.keys():
        data = locations[name]
        lat, lon = data["coords"]
        srs = data["SRs"]
        sr_low = srs[idx_low]
        sr_high = srs[idx_high]

        slope_low = (np.log(srs[idx_low + 1]) - np.log(srs[idx_low])) / (np.log(area_km2[idx_low + 1]) - np.log(area_km2[idx_low]))
        slope_high = (np.log(srs[idx_high + 1]) - np.log(srs[idx_high])) / (np.log(area_km2[idx_high + 1]) - np.log(area_km2[idx_high]))

        rows.append(
            {
                "Location": label_map.get(name, name),
                "Lat": lat,
                "Lon": lon,
                "SR_low": format_mean_std(float(np.mean(sr_low)), float(np.std(sr_low))),
                "SR_high": format_mean_std(float(np.mean(sr_high)), float(np.std(sr_high))),
                "Slope_low": format_mean_std(float(np.mean(slope_low)), float(np.std(slope_low))),
                "Slope_high": format_mean_std(float(np.mean(slope_high)), float(np.std(slope_high))),
            }
        )

    header = (
        "\\begin{table}\n"
        "    \\centering\n"
        "    \\small\n"
        "    \\setlength{\\tabcolsep}{5pt}\n"
        "    \\begin{tabularx}{\\textwidth}{l r r >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X}\n"
        "    \\toprule\n"
        "    Location & Lat & Lon & SR at $A_{low}$ & SR at $A_{high}$ & Slope at $A_{low}$ & Slope at $A_{high}$ \\\\\n"
        "    \\midrule\n"
    )

    body = ""
    for row in rows:
            body += (
                f"    {row['Location']} & {row['Lat']:.2f} & {row['Lon']:.2f} & {row['SR_low']} & {row['SR_high']} & {row['Slope_low']} & {row['Slope_high']} \\\\\n"
        )

    footer = (
        "    \\bottomrule\n"
        "    \\end{tabularx}\n"
        "    \\caption{SAR summary for $A_{low}=5\\times10^3\\,\\mathrm{m}$ and $A_{high}=5\\times10^4\\,\\mathrm{m}$ (areas in km$^2$). Values are mean ± SD across ensemble members.}\n"
        "    \\label{tab:sar_summary}\n"
        "\\end{table}\n"
    )

    output_path.write_text(header + body + footer)

    
if __name__ == "__main__":
    # creating X_maps for different resolutions
    seed = 1
    output_dir = Path("SARs")
    output_dir.mkdir(parents=True, exist_ok=True)


    model = load_ensemble_from_folds(RUN_DIR, device="cpu")

    env_ds, lc_ds, res_env_pixel = load_environmental_features(model)

    
    # dict_SAR = {"loc1": {"coords": (45.1, 6.3), #lat, long # old lat long
    #                    "SRs": [],},
    #           "loc2": {"coords": (53, 8.4),
    #                    "SRs": [],},
    #         "loc3": {"coords": (42.1, -5),
    #                    "SRs": [],}
    #         }
    
    dict_SAR = {"loc1": {"coords": (49.66, 16.00), # Žďárské vrchy Protected Landscape Area
                       "SRs": [],},
              "loc2": {"coords": (46.67, 10.2), # Parc Naziunal Svizzer
                       "SRs": [],},
            "loc3": {"coords": (52.05, 6.02), #Nationaal Park Veluwezoom
                       "SRs": [],}
            }
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035")
    window_sizes = np.logspace(np.log10(2e3), np.log10(1e6), 100)
    for loc in dict_SAR:
        print(loc)
        y, x = transformer.transform(*dict_SAR[loc]["coords"])
        dict_SAR[loc]["coords_epsg_3035"] = (x, y)
        for window_size in window_sizes:
            # predictor compilation
            features = build_features_for_window(
                model,
                env_ds,
                lc_ds,
                x,
                y,
                window_size,
                res_env_pixel,
            )

            # predictions
            SRs = np.concatenate([m.predict_sr_tot(features) for m in model.models], axis=1)
            dict_SAR[loc]["SRs"].append(SRs)
            
            ## predictions FIXME: legacy code
            # feature_scaler = checkpoint["feature_scalers"][0]
            # target_scaler = checkpoint["target_scalers"][0]
            # X = features[["log_observed_area"] + model.feature_names].values
            # X = feature_scaler.transform(X)
            # with torch.no_grad():
            #     X = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
            #     ys = np.concatenate([m._predict_sr_tot(X[:, 1:]).cpu().numpy() for m in model.models], axis=1) # predicting asymptote, no need to feed log_observed_area
            #     SRs = target_scaler.inverse_transform(ys.T).T # inverse transform to get back to original scale
            # dict_SAR[loc]["SRs"].append(SRs[0])  # SRs[0] since we have only one sample
        
        # Convert to numpy array with shape (len(window_sizes), len(model.models))
        dict_SAR[loc]["SRs"] = np.concatenate(dict_SAR[loc]["SRs"], axis=0)
        # dict_SAR[loc]["SRs"] = np.array(dict_SAR[loc]["SRs"]) # FIXME: legacy code
            
    dict_SAR["log_area"] = np.log(window_sizes**2)
    
    fig, ax = plt.subplots()
    dict_plot = {"loc1": {"c":"tab:blue"}, "loc2": {"c":"tab:red"}, "loc3": {"c":"tab:purple"}}
    for loc in dict_plot:
        d = dict_SAR[loc]
        arg_plot = dict_plot[loc]
        ax.plot(np.exp(dict_SAR["log_area"]), d["SRs"], c=arg_plot["c"])
        # ax.fill_between(np.exp(dict_SAR["log_area"]), 
        #     np.array(d["SR"]) - np.array(d["std_SR"]), 
        #     np.array(d["SR"]) + np.array(d["std_SR"]), 
        #     color=arg_plot["c"],
        #     alpha=0.4)
    ax.set_xscale("log")
    # ax.set_yscale("log")
    fig.savefig(output_dir / "SARs.pdf", dpi=300, bbox_inches="tight")
    save_to_pickle(output_dir / "SARs.pkl", dict_SAR=dict_SAR)
    export_sar_table(dict_SAR, window_sizes, output_dir / "SARs_table.tex")
