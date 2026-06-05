# MuScaRi Agent Guide

This repository contains the research code and manuscript sources for the
MuScaRi Nature Communications revision. Agents should preserve the existing
style of the project: concise, explicit research code in the style of a Google
DeepMind research scientist. Prefer readable scripts with visible constants over
large frameworks, hidden configuration layers, or clever abstractions.

## Execution
- Always use `uv` environment

## Project Map

- `muscari/`: core model, dataset, trainer, ensemble, utilities, and
  data-processing code. Put shared helper code in the relevant source module
  here, especially `muscari/utils.py` for cross-script utilities.
- `scripts/`: data processing, training, benchmarking, and projection scripts.
- `scripts/experiments_3rd_rev/`: lightweight third-revision experiment scripts. Keep
  hyperparameters as top-level constants.
- `figures/`: scripts used to generate main and supplementary figures.
- `paper/main.tex`: manuscript source. Match its technical style and notation.
- `paper/response_to_reviewers.md`: style reference for reviewer responses.
- `paper/response_to_reviewers_3rd_rev.md`: active third-revision response.
- `agent_log.yaml`: shared task and experiment log. Update it whenever a task or
  experiment starts or ends.

## Current Data And Result IDs

- Main local 1 km spatial-block dataset: `ceacce0`.
- Planned 100 km spatial-block dataset: `d0848f6` if available. If it is not
  available locally, generate a new 100 km dataset from the existing data
  processing scripts and record the generated ID and block size in
  `agent_log.yaml`. Do not add a dedicated third-revision compilation script
  unless the user asks for one.
- Canonical local GIFT evaluation dataset: `data/processed/test_samples_GIFT/418c563/compiled_data.parquet`,
  unless a newer compiled equivalent is deliberately regenerated and logged.
- Existing result paths are often hard-coded. Keep this explicit style, but make
  sure constants at the top of scripts are correct before launching runs.

## Coding Style

- Make minimal, local changes. Do not introduce a complex CLI, global experiment
  framework, or broad refactor unless the user explicitly asks for it.
- Apply the YAGNI principle
- Prefer small scripts copied or adapted from existing scripts over generic
  runners.
- Keep in-place hyperparameters visible near the top of each experiment script.
- Keep reusable utilities in `muscari/`, using existing modules when possible;
  experiment directories should contain runnable scripts, not shared source
  modules.
- Keep experiment-specific constants, such as selected bioclimatic variables,
  in the script that owns the experiment rather than in shared utilities.
- Preserve current stdout/checkpoint behavior and use `run_script.sh` for long
  jobs.
- Add succinct comments only where they prevent confusion.
- Do not reformat unrelated files.

## Commit And Pause Protocol

- Proceed incrementally. After each important refactoring task, run the relevant
  tests or smoke checks, report the staged changes and checks to the user, and
  ask for verification before committing. Commit only after the user approves.
- Keep commits narrow and descriptive. Do not include unrelated dirty worktree
  changes, pre-existing untracked directories, or generated smoke-test outputs
  unless the user explicitly asks for them.
- Before asking to commit, report which checks were run and update `agent_log.yaml`
  with the task outcome and next action.
- If a check cannot be run, record the reason in `agent_log.yaml` and mention it
  clearly in the user-facing pause message.

## Wandb Conventions

- Wandb support is optional and must be disabled by default unless an experiment
  script explicitly sets `USE_WANDB = True`.
- Use project `muscari-third-revision`.
- Use group names that encode the experiment family and dataset, for example
  `architecture_screen_ceacce0` or `final_100km_<dataset_id>`.
- Required config fields: git hash, hostname, dataset ID, GIFT dataset ID, fold,
  feature set, architecture variant, effort transform, asymptote transform,
  model family, batch size, learning rate, epoch limit, and output paths.
- Required metrics: train/validation/test metrics, GIFT metrics when available,
  relative-bias summaries, and residual-bias slope against `log_sp_unit_area`.
- Always finish wandb runs after each fold or experiment to avoid stale runs.

## Manuscript And Response Style

- Manuscript prose must match `paper/main.tex`: precise, compact, and technical.
- Reviewer responses must match `paper/response_to_reviewers.md`: quote the
  reviewer, acknowledge the point, state the change, and cite manuscript
  locations.
- Use placeholder locations such as `[MAIN: Results, Model performance]` until
  final line numbers are frozen.
- The reviewers are broadly supportive of the approach and novelty. Responses
  should emphasize that the revision follows their requests carefully with
  minimal disruption.
- If an item cannot be completed by an agent, leave an inline marker at the
  exact response location: `TODO(agent): ...`.

## Required Logging

Before starting and after finishing any substantial task, update
`agent_log.yaml`. Record:

- experiment or task name,
- status (`not_started`, `running`, `completed`, `failed`),
- start/end dates,
- hostname and git hash,
- dataset IDs and important parameters,
- wandb group/run links when used,
- key finding, outcome, failure reason if any,
- next actions.

Substantial tasks consist in large code modification, breaking changes, large experiments, etc...
