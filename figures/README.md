# Figure generation

Run figure scripts from the repository root with `uv run python`. Inputs are
read from `data/processed/` and `scripts/results/`; generated manuscript copies
are written only by scripts that define an explicit `PAPER_FIGURE_PATH`.

The main-figure entry points are:

- `figure_1/panels.py` and `figure_1/src_vs_sar.py`: synthetic conceptual panels.
- `figure_3/figure_3.py`: fold-matched performance comparisons and EVA/GIFT
  prediction diagnostics for SBCV dataset `ceacce0` and GIFT dataset `418c563`.
- `figure_4/figure_4.py`: EVA scale-binned NRMSE for the three MuScaRi feature
  variants. `figure_4_GIFT_variant.py` produces the GIFT sensitivity analysis.
- `figure_5/calculate_SR_dSR.py`, `figure_5/calculate_sar.py`, then
  `figure_5/figure_5_panels.py`: spatial projections, local species-area curves,
  and final panel assembly. Set `FIGURE5_RUN_DIR` to override the recorded model
  artifact.

Supplementary scripts live under `SI/` and declare their input paths near the
top of each file. The retained `SI/convergence/convergence.pdf` is a historical
artifact: its legacy training-loss checkpoint no longer exists, so the broken
generator was removed during the pre-submission audit.
