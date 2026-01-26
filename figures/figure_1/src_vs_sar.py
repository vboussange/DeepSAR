"""Synthetic illustration linking species rarefaction curves to SAR."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def weibull4(log_x: np.ndarray, b: float, c: float, d: float, e: float) -> np.ndarray:
	"""4-parameter Weibull on log-scale input."""
	return c + (d - c) * np.exp(-np.exp(b * (log_x - np.log(e))))


def sar_power_law(area: np.ndarray, c0: float, z: float) -> np.ndarray:
	return c0 * np.power(area, z)


def build_rarefaction_curves(areas: list[float]) -> list[dict[str, np.ndarray]]:
	curves = []
	b = -2.2
	c = 0.0
	c0, z = 5.0, 0.25
	min_effort = 1e2
	max_frac = [0.055, 0.055, 0.010]

	for area, frac in zip(areas, max_frac):
		asymptote = sar_power_law(np.array([area]), c0, z)[0]
		e = min_effort * 2.0
		x = np.logspace(np.log10(min_effort), np.log10(np.max(areas)), 120)
		y = weibull4(np.log(x), b=b, c=c, d=asymptote, e=e)
		max_effort = area * frac
		curves.append(
			{
				"area": area,
				"x": x,
				"y": y,
				"asymptote": asymptote,
				"max_effort": max_effort,
			}
		)
	return curves


def plot_src_vs_sar(curves: list[dict[str, np.ndarray]]) -> None:
	rng = np.random.default_rng(42)
	colors = ["#f72585", "#4cc9f0", "#3a0ca3"]
	areas = np.array([c["area"] for c in curves])

	fig, (ax_src, ax_sar) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

	for curve, color in zip(curves, colors):
		area = curve["area"]
		x = curve["x"]
		y = curve["y"]
		max_effort = curve["max_effort"]

		mask_obs = x <= max_effort
		mask_extrap = x >= max_effort

		ax_src.plot(x[mask_obs], y[mask_obs], color=color, linewidth=2)
		ax_src.plot(x[mask_extrap], y[mask_extrap], color=color, linewidth=2, linestyle="--")

		obs_x = np.logspace(np.log10(x[0]), np.log10(max_effort), 20)
		obs_y = weibull4(np.log(obs_x), b=-2.2, c=0.0, d=curve["asymptote"], e=2e2)
		logx = np.log10(obs_x)
		noise_std = np.interp(logx, [logx.min(), logx.max()], [0.35, 0.05])
		noise = rng.normal(0.0, noise_std)
		obs_y_noisy = np.clip(obs_y * (1.0 + noise), 1e-3, None)

		ax_src.scatter(
			obs_x,
			obs_y_noisy,
			color=color,
			s=18,
			alpha=0.8,
			label=f"A = {area:,.0f} m²",
		)

	ax_src.set_xscale("log")
	ax_src.set_ylabel("Species richness")
	ax_src.set_xlabel("Sampling effort (m²)")
	ax_src.set_title("Species rarefaction curves")
	ax_src.grid(True, alpha=0.3)
	ax_src.legend(frameon=False, fontsize=9)

	asymptotes = np.array([c["asymptote"] for c in curves])
	area_grid = np.logspace(np.log10(1e2), np.log10(1e6), 200)
	sar_line = sar_power_law(area_grid, c0=5.0, z=0.25)

	ax_sar.plot(area_grid, sar_line, color="#222222", linewidth=2, label="")
	ax_sar.scatter(areas, asymptotes, color=colors, s=30, zorder=3)
	ax_sar.set_xscale("log")
	# ax_sar.set_yscale("log")
	ax_sar.set_xlabel("Spatial unit area (m²)")
	# ax_sar.set_ylabel("")
	ax_sar.set_title("Species–area relationship")
	ax_sar.grid(True, alpha=0.3)
	ax_sar.legend(frameon=False, fontsize=9)

	fig.tight_layout()
	fig.savefig(Path(__file__).with_suffix(".pdf"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
	spatial_areas = [1e4, 3e4, 1e5]
	curves = build_rarefaction_curves(spatial_areas)
	plot_src_vs_sar(curves)
