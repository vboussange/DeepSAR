# Current Sub Plan

## Summary

- Use `dev_3rd_rev` for code, data, benchmark, and logging commits.
- Use a separate writing branch for manuscript and response changes.
- Use external read-only subagents after major changes, asking them to critique the work as paper reviewers.
- Never add em dashes in new writing.

## Phase 0: Branch And State Check

- Confirm clean worktree on `dev_3rd_rev`.
- Record task start in `agent_log.yaml`.
- Confirm whether `d0848f6` exists locally.
- Confirm `paper/` submodule branch state before writing work.

Commit:
- Commit `CURRENT_SUB_PLAN.md` and the start log entry on `dev_3rd_rev`.

## Phase 1: Fix Log-Linear Benchmark

- Patch `scripts/benchmark.py` so the Ridge log-linear baseline writes full metric schema:
  - `train_*`, `val_*`, `test_*`, `gift_*`
  - `interp_* = test_*`
  - `extrap_* = gift_*`
- Keep the model definition unchanged:
  - Ridge on `log1p(sr)`
  - `StandardScaler`
  - validation-selected alpha
  - `expm1` inverse
  - non-negative clipping only
- Add diagnostics for alpha, sample counts, finite predictions, and GIFT prediction range.
- Run smoke checks with `uv`.

Subagent:
- Spawn a read-only benchmark audit subagent to check leakage risk, metric consistency, and fairness of the linear baseline.

Commit:
- Commit the log-linear fix after smoke checks and audit response.

## Phase 2: Prepare 100 km SBCV Dataset

- If `d0848f6` exists and is complete, use it.
- If absent, use the existing SBCV compiler with a minimal explicit block-size override:
  - `MUSCARI_SBCV_BLOCK_SIZE_M=100000`
- Do not add a dedicated 100 km compilation script.
- Verify:
  - 5 train parquet files
  - 5 validation parquet files
  - 5 test parquet files
  - metadata records `block_size_m = 100000`
- Record dataset ID and parameters in `agent_log.yaml`.

Commit:
- Commit the dataset-generation interface and logging updates before launching long work.
- Commit completed dataset metadata and final log update after generation.

## Phase 3: Run 100 km Benchmark

- Make `scripts/benchmark.py` accept `MUSCARI_SBCV_ID=<dataset_id>` while keeping `ceacce0` as default.
- Run selected architecture on the 100 km SBCV dataset:
  - `MuScaRi_Area`
  - `MuScaRi_ClimateDEM`
  - `MuScaRi_ClimateDEM_Area`
  - `FFNN_ClimateDEM_Area`
  - corrected `Linear_ClimateDEM_Area`
- Use `run_script.sh` for long benchmark runs.
- Write results to `scripts/results/benchmark/benchmark_results_<100km_id>.csv`.
- Validate row count and missing metric columns.

Subagent:
- Spawn a read-only results audit subagent to assess robustness interpretation and benchmark completeness.

Commit:
- Commit completed 100 km benchmark results and `agent_log.yaml` outcome.

## Phase 4: Results Synthesis

- Summarize final numbers for:
  - 1 km SBCV benchmark
  - 100 km SBCV robustness benchmark
  - corrected linear baseline
  - Figure 4 scale-binned EVA result
  - GIFT asymptotic audit
- Decide which values enter main text versus SI.
- Keep interpretation predictive and diagnostic, not mechanistic.

Commit:
- Commit synthesis tables or summaries if new tracked artifacts are created.

## Phase 5: Writing Branch Setup

- Create a separate superproject branch from updated `dev_3rd_rev`, for example `paper_3rd_revision_writing`.
- Inside `paper/`, create a matching branch from `3rd_revision`.
- Make all manuscript and reviewer-response edits on this writing branch.

Commit:
- Commit only writing-related changes on the writing branch.

## Phase 6: Manuscript Updates

- Update `paper/main.tex` to:
  - Replace Shapley-based Figure 4 interpretation with scale-binned relative RMSE.
  - Add corrected log-linear baseline.
  - Add 100 km SBCV robustness result.
  - Update SI performance discussion and tables where needed.
- Match the existing manuscript style: compact, technical, cautious.
- Check that new text contains no em dashes.

Subagent:
- Spawn a read-only manuscript audit subagent as a skeptical reviewer.

Commit:
- Commit manuscript updates inside `paper/`.
- Commit the `paper` submodule pointer in the superproject branch.

## Phase 7: Reviewer Response Updates

- Update `paper/response_to_reviewers_3rd_rev.md`.
- For each relevant reviewer point:
  - quote the concern
  - acknowledge it
  - state the change
  - cite placeholder manuscript locations
- Emphasize minimal, reviewer-driven revisions.
- Keep unresolved human decisions marked as `TODO(agent): ...`.

Subagent:
- Spawn a read-only response audit subagent as Reviewers 1 and 3.

Commit:
- Commit reviewer-response updates inside `paper/`.
- Commit the final `paper` submodule pointer and `agent_log.yaml` update in the superproject branch.

## Final Checks

- `uv run python -m py_compile scripts/benchmark.py scripts/data_processing/compile_sbcv_eva_samples.py`
- `MUSCARI_SMOKE_TEST=1 MUSCARI_USE_WANDB=0 uv run python scripts/benchmark.py`
- Validate final benchmark CSVs for missing primary metrics.
- Compile the paper if the local TeX environment supports it.
- `git diff --check`
- `git diff | python -c 'import sys; s=sys.stdin.read(); raise SystemExit(1 if chr(0x2014) in s else 0)'` must pass.

## Assumptions

- `dev_2rd_rev` meant `dev_3rd_rev`.
- The log-linear model remains a faithful baseline even if GIFT extrapolation remains poor.
- If `d0848f6` is unavailable, the regenerated 100 km dataset ID will be recorded and used consistently.
- Writing changes belong on a separate branch, including commits inside the `paper` submodule.
