# MuScaRi Third-Revision Plan

## Summary

- Work in three ordered phases: agent/logging files first, experiments second, manuscript and response revision third.
- Keep code changes minimal: lightweight experiment scripts with in-place constants, no complex CLI.
- Add wandb as a thin experiment-tracking layer.
- In the response, emphasize that reviewers are broadly positive about the approach and novelty, and that revisions follow their wording/analysis requests with minimal disruption.

## Phase 1: Agent And Logging Files
**STATUS**: DONE ✔️

- Create root `AGENT.md` before any experiment or manuscript task.
  - Document project structure, important paths, current dataset IDs, and expected execution style.
  - State that code should follow the repo’s existing style: concise, explicit research code in the style of a Google DeepMind research scientist.
  - State that manuscript prose must match `paper/main.tex`, and reviewer responses must match `paper/response_to_reviewers.md`.
  - Instruct agents to avoid broad abstractions, heavy CLIs, unrelated refactors, and formatting churn.
  - Document wandb conventions: project name, group naming, tags, required metadata, and how to disable tracking.
  - Require unresolved human-only items to be marked inline in the response draft as `TODO(agent): ...`.

- Create root `agent_log.yaml`.
  - Track `active_experiments`, `completed_experiments`, `failed_experiments`, `next_actions`, and `notes`.
  - Record experiment name, dataset ID, block size, architecture variant, feature set, wandb group/run link, hostname, git hash, start/end time, key finding, and outcome.
  - Update this file whenever an experiment or manuscript task starts or ends.

## Phase 2: Experiments
**STATUS**: WIP

- Add minimal wandb support. ✅
  - Add `wandb` dependency.
  - Add `USE_WANDB`, `WANDB_PROJECT`, `WANDB_GROUP`, and `WANDB_TAGS` constants directly in experiment scripts.
  - Use PyTorch Lightning `WandbLogger` only when `USE_WANDB=True`.
  - Log fold ID, dataset ID, feature set, architecture variant, train/val/test metrics, GIFT metrics, git hash, and output paths.
  - Preserve current local stdout/checkpoint behavior.

- Keep experiment code lightweight. ✅
  - Do not introduce a CLI or centralized experiment framework.
  - Prefer small scripts under `scripts/experiments/` copied/adapted from current `train.py`, `benchmark.py`, and `chao2_estimator.py`.
  - Keep hyperparameters as visible constants at the top of each script.
  - Use `run_script.sh` and `stdout/` for long-running jobs.

- Run private architecture screen first on `ceacce0`. 🟡
  - `current_abs`: current MuScaRi, absolute effort.
  - `exp_abs`: exponential/safe-positive asymptote parameter, absolute effort.
  - `current_rel`: current asymptote, relative effort `log_observed_area - log_sp_unit_area`.
  - `exp_rel`: exponential/safe-positive asymptote plus relative effort.
  - Use only `env + area` features.
  - Select architecture by GIFT RMSE, interpolation RMSE, and residual-bias slope versus `log_sp_unit_area`; ties within 2% prefer the simpler variant.

- Run final manuscript experiments after architecture selection. **TODO**
  - Use 1 km block dataset `ceacce0`.
  - Use 100 km block dataset `d0848f6` if available; otherwise generate a new 100 km SBCV dataset and record its ID in `agent_log.yaml`.
  - Run MuScaRi `area`, MuScaRi `env`, MuScaRi `env + area`, FFNN `env + area`, regularized log-linear `env + area`, and Chao2 for GIFT extrapolation only.
  - Linear baseline: Ridge on `log1p(sr)`, validation-selected alpha, `expm1` inverse transform, same fold/evaluation protocol.

## Phase 3: Manuscript And Response Revision

- First update manuscript sections affected by the new runs and architecture.
  - Update Methods for the selected architecture, effort representation, asymptote transform if selected, wandb-free reproducibility details, and final datasets.
  - Update Results, Figure 3, SI performance tables, and 100 km robustness output.
  - Recompute downstream selected-model outputs only after final architecture is fixed.
  - Use placeholders in response text such as `[MAIN: Results, Model performance]` until final line numbers are frozen.

- Then address reviewer comments one at a time, following their recipes with minimal changes.
  - Start the response by noting that reviewers were broadly supportive of the approach and novelty, and that the revision mainly clarifies wording, interpretation, and validation.
  - For each comment: acknowledge, state the precise change, and point to placeholder manuscript locations.
  - Avoid arguing against reviewer framing unless strictly necessary.
  - When a comment asks for caution, soften the manuscript language rather than defending the stronger claim.

- Manuscript content priorities:
  - Reframe abstract/introduction toward explicit sampling-effort modeling while keeping SARs as derived outputs.
  - Clarify rarefaction parameters, the asymptote/total-richness parameter, neural-network motivation, `T`, and ensemble terminology.
  - Add target-richness distribution context so RMSE is interpretable.
  - Explain similar interpolation performance without area using feature correlations.
  - Add the linear baseline and 100 km spatial-block robustness result.
  - Moderate Shapley and mechanism language: replace “disentangle” and causal claims with model-based attribution language.
  - Clarify top-down richness-model literature and the distinction from Andermann et al. 2022.
  - Update Figure 1 nesting/order and Figure 5 grey-block caption or styling.

## Test Plan And Assumptions

- Smoke-test each new experiment script with one fold and one epoch.
- Verify wandb can be disabled cleanly.
- Verify checkpoints load, fold metrics are finite, predictions are non-negative, and rarefaction curves remain monotonic.
- Verify manuscript figures/tables regenerate and `paper/main.tex` compiles.
- Assumptions fixed: focused four-run architecture screen, 1 km plus 100 km final spatial blocks, regularized linear baseline, minimal code changes, lightweight scripts, and wandb tracking.
