"""Generate final Figure 5 without manual layout correction."""

import os
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import xarray as xr
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from muscari.plotting import CMAP_BR, CMAP_DSR

ROOT = Path(__file__).parents[2]
MODEL_NAME = os.environ.get("FIGURE5_MODEL_NAME", "dae0789a3c87")

CONFIG = {
    "model_name": MODEL_NAME,
    "sar_path": Path(__file__).parent / "SARs",
    "rolling_kwargs": {"x": 2, "y": 2, "center": False, "min_periods": 2},
    "figure_size": (8, 10),
    "sar_window": 10,
    "sar_xlim": (2e3**2 / 1e6, 2e5**2 / 1e6),
    "sar_ylim": (500, 2200),
    "sar_vlines": [5e3**2 / 1e6, 5e4**2 / 1e6],
    "locations": {
        "loc1": {
            "label": "A",
            "color": "tab:blue",
            "title": "Žďárské vrchy P. L. A.\n(Czech Republic)",
            "callout": {"5000": (0.93, 0.54), "50000": (0.93, 0.52)},
        },
        "loc2": {
            "label": "B",
            "color": "tab:red",
            "title": "Parc Naziunal Svizzer\n(Switzerland)",
            "callout": {"5000": (0.38, -0.03), "50000": (0.38, -0.03)},
        },
        "loc3": {
            "label": "C",
            "color": "tab:purple",
            "title": "Nationaal Park Veluwezoom\n(Netherlands)",
            "callout": {"5000": (0.22, 0.72), "50000": (0.24, 0.70)},
        },
    },
    "map_panels": [
        {
            "key": "sr_5000",
            "prefix": "SR",
            "res_m": 5000,
            "letter": "a",
            "title": "5 $\\times$ 5 km$^2$",
            "cbar_loc": "left",
            "cbar_label": "Predicted species\nrichness ($S_T$)",
            "cmap": CMAP_BR,
            "show_callouts": True,
        },
        {
            "key": "sr_50000",
            "prefix": "SR",
            "res_m": 50000,
            "letter": "b",
            "title": "50 $\\times$ 50 km$^2$",
            "cbar_loc": "right",
            "cbar_label": "Predicted species\nrichness ($S_T$)",
            "cmap": CMAP_BR,
            "show_callouts": True,
        },
        {
            "key": "dS_dA_5000",
            "prefix": "dS_dA",
            "res_m": 5000,
            "letter": "c",
            "title": "",
            "cbar_loc": "left",
            "cbar_label": "Species accumulation\nrate ($dS_T/dA$)\n(species km$^{-2}$)",
            "cmap": CMAP_DSR,
            "show_callouts": False,
        },
        {
            "key": "dS_dA_50000",
            "prefix": "dS_dA",
            "res_m": 50000,
            "letter": "d",
            "title": "",
            "cbar_loc": "right",
            "cbar_label": "Species accumulation\nrate ($dS_T/dA$)\n(species km$^{-2}$)",
            "cmap": CMAP_DSR,
            "show_callouts": False,
        },
    ],
}

RAST_PATH = ROOT / "data" / "processed" / "projections" / CONFIG["model_name"]

plt.rcParams.update({
    "font.size": 9,
    "font.weight": "normal",
    "axes.titlesize": 14,
    "axes.titleweight": "normal",
    "axes.labelsize": 13,
    "axes.labelweight": "normal",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 16,
    "lines.markersize": 3,
})


def raster_name(prefix: str, res_m: int) -> str:
    return f"{prefix}_raster_{CONFIG['model_name']}_{int(res_m)}m"


def load_data() -> tuple[dict[str, xr.DataArray], dict]:
    rasters = {}
    for path in RAST_PATH.glob("*.tif"):
        rasters[path.stem] = rioxarray.open_rasterio(path).squeeze(drop=True)

    with open(CONFIG["sar_path"] / "SARs.pkl", "rb") as pickle_file:
        dict_sar = pickle.load(pickle_file)["dict_SAR"]
    return rasters, dict_sar


def load_europe_geometry():
    world = gpd.read_file(ROOT / "data" / "raw" / "NaturalEarth" / "ne_10m_admin_0_countries.shp")
    return world[world.CONTINENT == "Europe"].to_crs("EPSG:3035").geometry


def prepared_rasters(raw_rasters: dict[str, xr.DataArray], europe_geom) -> dict[str, xr.DataArray]:
    rasters = {}
    for panel in CONFIG["map_panels"]:
        name = raster_name(panel["prefix"], panel["res_m"])
        if name not in raw_rasters:
            raise FileNotFoundError(f"Missing raster: {name}")
        raster = (
            raw_rasters[name]
            .rolling(**CONFIG["rolling_kwargs"])
            .mean()
            .rio.clip(europe_geom, drop=True)
            .rio.write_crs("EPSG:3035")
        )
        if panel["prefix"] == "dS_dA":
            raster = xr.where((raster >= 0) | (~np.isfinite(raster)), raster, 0)
        rasters[panel["key"]] = raster
    return rasters


def add_colorbar(fig, ax, artist, panel):
    if panel["cbar_loc"] == "left":
        cax = inset_axes(
            ax,
            width="2.4%",
            height="54%",
            loc="center left",
            bbox_to_anchor=(-0.06, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(artist, cax=cax)
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.yaxis.set_ticks_position("left")
    else:
        cax = inset_axes(
            ax,
            width="2.4%",
            height="54%",
            loc="center right",
            bbox_to_anchor=(0.06, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(artist, cax=cax)
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.yaxis.set_ticks_position("right")
    cbar.set_label(panel["cbar_label"], fontsize=11, labelpad=3)
    cbar.ax.tick_params(labelsize=8, width=1.0, length=3)
    return cbar


def plot_map(fig, ax, raster, panel, dict_sar):
    vmin = float(raster.quantile(0.01))
    vmax = float(raster.quantile(0.99))
    artist = raster.plot(
        ax=ax,
        cmap=panel["cmap"],
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        add_labels=False,
    )
    artist.set_rasterized(True)
    ax.set_axis_off()
    ax.set_aspect("equal")
    if panel["title"]:
        ax.set_title(panel["title"], fontsize=14, pad=8)
    ax.text(-0.08, 0.86, panel["letter"], transform=ax.transAxes, fontsize=16, fontweight="bold")
    add_colorbar(fig, ax, artist, panel)
    if panel["show_callouts"]:
        add_callouts(ax, dict_sar, panel["res_m"])


def add_callouts(ax, dict_sar, res_m):
    res_key = str(int(res_m))
    for loc, cfg in CONFIG["locations"].items():
        x, y = dict_sar[loc]["coords_epsg_3035"]
        color = cfg["color"]
        label = f"{cfg['label']}{1 if res_m == 5000 else 2}"
        ax.scatter(
            [x],
            [y],
            marker="s",
            s=72 if res_m == 5000 else 95,
            facecolor=color,
            edgecolor=color,
            linewidth=2,
            alpha=0.45,
            zorder=4,
        )
        ax.annotate(
            label,
            xy=(x, y),
            xytext=cfg["callout"][res_key],
            xycoords="data",
            textcoords=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.2, "shrinkA": 2, "shrinkB": 4},
            clip_on=False,
            zorder=5,
        )


def smooth_curve(values, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode="valid")


def plot_sar_panel(ax, loc, dict_sar, show_ylabel=False):
    cfg = CONFIG["locations"][loc]
    area = np.exp(dict_sar["log_area"]) / 1e6
    sr = np.asarray(dict_sar[loc]["SR"], dtype=float)
    std = np.asarray(dict_sar[loc]["std_SR"], dtype=float)
    window = CONFIG["sar_window"]

    area_smooth = area[window - 1 :]
    sr_smooth = smooth_curve(sr, window)
    low_smooth = smooth_curve(sr - std, window)
    high_smooth = smooth_curve(sr + std, window)

    ax.plot(area_smooth, sr_smooth, color=cfg["color"], linewidth=2)
    ax.fill_between(area_smooth, low_smooth, high_smooth, color=cfg["color"], alpha=0.28, linewidth=0)
    for idx, xline in enumerate(CONFIG["sar_vlines"], start=1):
        ax.axvline(x=xline, color="0.45", linestyle="--", linewidth=1.4, alpha=0.8)
        ax.text(
            xline,
            1.07,
            f"{cfg['label']}{idx}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )

    ax.set_xscale("log")
    ax.set_xlim(*CONFIG["sar_xlim"])
    ax.set_ylim(*CONFIG["sar_ylim"])
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="0.6", alpha=0.7)
    ax.set_title(cfg["title"], fontsize=10, pad=34)
    if show_ylabel:
        ax.set_ylabel("Predicted species\nrichness ($S_T$)", fontsize=11)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=8, width=1.0, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def save_figure(fig):
    outputs = [
        Path("figure_5_panels.pdf"),
        Path("figure_5_panels.png"),
        Path("figure_5_panels.svg"),
        ROOT / "paper" / "figures" / "figure_5.pdf",
        ROOT / "paper" / "figures" / "figure_5.png",
        ROOT / "paper" / "figures" / "figure_5.svg",
    ]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, facecolor="white")


def main():
    raw_rasters, dict_sar = load_data()
    rasters = prepared_rasters(raw_rasters, load_europe_geometry())

    fig = plt.figure(figsize=CONFIG["figure_size"], facecolor="white")
    map_grid = GridSpec(
        2,
        2,
        figure=fig,
        left=0.14,
        right=0.86,
        top=0.96,
        bottom=0.305,
        wspace=0.12,
        hspace=0.16,
    )

    map_axes = [
        fig.add_subplot(map_grid[0, 0]),
        fig.add_subplot(map_grid[0, 1]),
        fig.add_subplot(map_grid[1, 0]),
        fig.add_subplot(map_grid[1, 1]),
    ]
    for ax, panel in zip(map_axes, CONFIG["map_panels"]):
        plot_map(fig, ax, rasters[panel["key"]], panel, dict_sar)

    sar_grid = GridSpec(
        1,
        3,
        figure=fig,
        left=0.14,
        right=0.94,
        bottom=0.09,
        top=0.22,
        wspace=0.09,
    )
    sar_axes = [fig.add_subplot(sar_grid[0, i]) for i in range(3)]
    for i, loc in enumerate(CONFIG["locations"]):
        plot_sar_panel(sar_axes[i], loc, dict_sar, show_ylabel=(i == 0))

    sar_axes[0].text(-0.13, 1.17, "e", transform=sar_axes[0].transAxes, fontsize=16, fontweight="bold")
    fig.text(0.54, 0.035, "Spatial unit area, $A$ (km$^2$)", ha="center", fontsize=14)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
