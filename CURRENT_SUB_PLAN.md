# Current Sub Plan

## Status Snapshot

- [x] Code, benchmark, figure, manuscript, and response work for this sub-plan are complete.
- [x] Work was moved to the separate writing branch `paper_3rd_revision_writing`.
- [x] Major steps were committed incrementally.
- [x] External read-only audit subagents reviewed benchmark, results, manuscript, and response changes.
- [x] Final supported checks passed.
- [ ] Local TeX compilation remains unchecked because no TeX engine is installed on this machine.
- [ ] Hand-built figure tasks remain for the user before submission.

## Completed Work

### Phase 0: Branch And State Check

- [x] Confirmed the starting branch and worktree state.
- [x] Recorded the task in `agent_log.yaml`.
- [x] Confirmed the local 100 km SBCV dataset `d0848f6` exists and is complete.
- [x] Confirmed the `paper/` submodule branch state.
- [x] Committed the sub-plan and initial log entry.

Commits:
- [x] Superproject: `93f1599 Record final revision sub-plan`

### Phase 1: Fix Log-Linear Benchmark

- [x] Patched `scripts/benchmark.py` so `Linear_ClimateDEM_Area` writes the full benchmark metric schema.
- [x] Kept the model definition unchanged: Ridge on `log1p(sr)`, `StandardScaler`, validation-selected alpha, `expm1`, and non-negative clipping.
- [x] Added diagnostics for alpha, split sample counts, finite predictions, GIFT prediction range, and residual-bias slopes.
- [x] Added `MUSCARI_USE_WANDB` and `MUSCARI_SBCV_ID` overrides while keeping defaults stable.
- [x] Ran smoke checks with `uv`.
- [x] Spawned a read-only benchmark audit subagent.

Commits:
- [x] Superproject: `4abe164 Fix log-linear benchmark metrics`

### Phase 2: Prepare 100 km SBCV Dataset

- [x] Verified `data/processed/training_samples/sbcv/d0848f6`.
- [x] Confirmed 5 train, 5 validation, and 5 test parquet files.
- [x] Confirmed `config_used.json` records `block_size=100000`.
- [x] Recorded the dataset ID and parameters in `agent_log.yaml`.
- [x] Did not add a dedicated third-revision compilation script.

Commits:
- [x] Superproject: `23b90ee Record 100km SBCV dataset availability`

### Phase 3: Run 100 km Benchmark

- [x] Ran the selected architecture benchmark on `d0848f6`.
- [x] Included `MuScaRi_Area`.
- [x] Included `MuScaRi_ClimateDEM`.
- [x] Included `MuScaRi_ClimateDEM_Area`.
- [x] Included `FFNN_ClimateDEM_Area`.
- [x] Included corrected `Linear_ClimateDEM_Area`.
- [x] Wrote `scripts/results/benchmark/benchmark_results_d0848f6.csv`.
- [x] Validated 25 rows, 5 folds per model, and no missing primary metrics.
- [x] Spawned a read-only results audit subagent.

Commits:
- [x] Superproject: `458cb57 Record 100km benchmark launch`
- [x] Superproject: `478b000 Add 100km SBCV benchmark results`

### Phase 4: Results Synthesis

- [x] Corrected the 1 km benchmark linear rows in `benchmark_results_ceacce0.csv`.
- [x] Summarized 1 km SBCV benchmark results.
- [x] Summarized 100 km SBCV robustness results.
- [x] Summarized the corrected log-linear baseline.
- [x] Summarized Figure 4 scale-binned EVA results.
- [x] Summarized the GIFT asymptotic audit.
- [x] Wrote `scripts/results/final_revision_synthesis.md`.
- [x] Kept interpretation predictive and diagnostic, not mechanistic.

Commits:
- [x] Superproject: `2a2f02a Synthesize final revision benchmark results`

### Phase 5: Writing Branch Setup

- [x] Created the superproject writing branch `paper_3rd_revision_writing`.
- [x] Created the matching `paper/` submodule branch `paper_3rd_revision_writing`.
- [x] Kept manuscript and response edits on the writing branch.

Commits:
- [x] Superproject: `0ccf982 Record writing branch setup`

### Phase 6: Manuscript And Figure Updates

- [x] Updated `paper/main.tex` with the final benchmark narrative.
- [x] Replaced Shapley-based Figure 4 interpretation with scale-binned relative RMSE.
- [x] Added the corrected log-linear baseline to the manuscript and SI performance tables.
- [x] Added the 100 km SBCV robustness result.
- [x] Recomputed Chao2 outputs and included Chao2 in Figure 3.
- [x] Regenerated Figure 3 with corrected benchmark rows, the log-linear baseline, Chao2, and ClimateDEM prediction panels.
- [x] Fixed the Figure 3 panel c checkpoint-loading bug by using `MuScaRi.initialize(...)`.
- [x] Copied regenerated Figure 3 and Figure 4 assets into `paper/figures/`.
- [x] Updated Figure 5 manuscript text to state that maps use the ClimateDEM-only ensemble.
- [x] Checked new writing for em dashes.
- [x] Spawned a read-only manuscript audit subagent and addressed blocking findings.

Commits:
- [x] Paper submodule: `c73fd07 Update manuscript benchmark narrative`
- [x] Paper submodule: `045d9e8 Fix Figure 3 benchmark panel`
- [x] Superproject: `1d7de8b Update manuscript figures and Chao2 baseline`
- [x] Superproject: `3c2314c Fix Figure 3 checkpoint loading`

Key check:
- [x] Figure 3 panel c audited after the fix: RMSE `45.95`, R2 `0.988`, median relative bias `0.021`.

### Phase 7: Reviewer Response Updates

- [x] Updated `paper/response_to_reviewers_3rd_rev.md`.
- [x] Addressed the 100 km spatial-block robustness point.
- [x] Addressed the corrected log-linear baseline.
- [x] Addressed the Shapley replacement and Figure 4 predictive framing.
- [x] Addressed the RMSE scale context.
- [x] Addressed ClimateDEM-only mapping text for Figure 5.
- [x] Added main-text discussion of direct macroecological models.
- [x] Added a limitation on absolute versus relative sampling effort.
- [x] Clarified nested spatial units in the Figure 1 caption.
- [x] Clarified grey grouping blocks in the Figure 5 caption.
- [x] Removed hidden HTML comments from the response draft.
- [x] Spawned a read-only response audit subagent and addressed blocking findings.

Commits:
- [x] Paper submodule: `e5b1d54 Update third revision reviewer responses`
- [x] Superproject: `c155895 Record reviewer response updates`

### Phase 8: Final Checks

- [x] Ran `uv run python -m py_compile scripts/benchmark.py scripts/data_processing/compile_sbcv_eva_samples.py scripts/chao2_estimator.py figures/figure_3/figure_3.py muscari/cld.py`.
- [x] Ran `MUSCARI_SMOKE_TEST=1 MUSCARI_USE_WANDB=0 uv run python scripts/benchmark.py`.
- [x] Validated `benchmark_results_ceacce0.csv`.
- [x] Validated `benchmark_results_d0848f6.csv`.
- [x] Validated `benchmark_chao2_results_ceacce0.csv`.
- [x] Validated the smoke benchmark CSV.
- [x] Ran `git diff --check`.
- [x] Ran `git -C paper diff --check`.
- [x] Checked current diffs for em dashes.
- [x] Checked that stale Shapley references and old percentages were removed from `paper/main.tex`.
- [x] Confirmed the superproject worktree was clean after final commits.
- [x] Confirmed the `paper/` submodule worktree was clean after final commits.
- [x] Recorded final checks in `agent_log.yaml`.

Commits:
- [x] Superproject: `59662fd Record final revision checks`

## Remaining Before Submission

- [ ] Reorder the top-panel illustrations in the hand-assembled Figure 1 source, then remove the visible `TODO(agent)` marker in `paper/response_to_reviewers_3rd_rev.md`.
- [ ] Hand-build the final Figure 5 image using the ClimateDEM-only ensemble, as planned by the user.
- [ ] Refresh any Figure 5 focal-location numeric summaries if the hand-built image changes values shown in the text or caption.
- [ ] Compile the paper on a machine with a TeX engine and inspect warnings, floats, references, and bibliography.
- [ ] Do one final human read-through of `paper/main.tex` and `paper/response_to_reviewers_3rd_rev.md`.
- [ ] Decide whether to merge or cherry-pick the writing branch back into the submission branch.

## Current Branch And Cleanliness

- [x] Superproject branch: `paper_3rd_revision_writing`.
- [x] Paper submodule branch: `paper_3rd_revision_writing`.
- [x] Superproject clean after final log commit.
- [x] Paper submodule clean after final response commit.

## Assumptions

- [x] `dev_2rd_rev` meant `dev_3rd_rev`.
- [x] The log-linear model remains a faithful baseline even though GIFT extrapolation remains poor.
- [x] `d0848f6` is the 100 km SBCV dataset used for robustness checks.
- [x] Writing changes belong on the separate writing branch, including commits inside the `paper/` submodule.
