# Repository Guidelines

## Project

`muscari/` contains the core Python package for models, datasets, plotting, training utilities, and `muscari/data_processing/` helpers. `scripts/` holds workflows for data processing, training, benchmarking, and projection. `widget/` is a separate Gradio app package (`muscari_widget/`) with its own metadata and tests in `widget/tests/`. `figures/` contains paper figure scripts, `paper/` contains the live Overleaf manuscript project for the third revision, `legacy_paper/` contains the previous manuscript working copy and non-Overleaf revision artifacts, and `data/` stores raw-data placeholders and local inputs.

The current project stage is revision, after receiving the second round of feedback from reviewers (see `/Users/victorboussange/projects/MuScaRi/paper/response_to_reviewers_3rd_rev.md`)

## Manuscript Repository Layout

`paper/` is now a standalone clone of the Overleaf v3 project (`https://git@git.overleaf.com/6a47cf0a15da9af5443c3709`) on branch `main`. Treat this as the live manuscript source that Overleaf compiles. Keep it curated: manuscript sources, bibliography, style file, referenced figures, `cover_letter.md`, `cover_letter.pdf`, and `response_to_reviewers_3rd_rev.md` are appropriate; LaTeX build outputs, `main.pdf`, `response_to_reviewers_3rd_rev.pdf`, and `revision_diff_*` files should not be committed there.

`legacy_paper/` is the old manuscript repository working copy, backed up on `git@github.com:vboussange/MuScaRi_manuscript.git` branch `paper_3rd_revision_writing`. It retains planning files, old submission PDFs, response PDFs, and previous LaTeX diff artifacts. Because it was renamed from `paper/`, its `.git` file still points at `.git/modules/paper`; if Git status is needed there, use an explicit work tree, for example:

```bash
git --git-dir=/Users/victorboussange/projects/MuScaRi/.git/modules/paper \
  --work-tree=/Users/victorboussange/projects/MuScaRi/legacy_paper status
```

To regenerate an up-to-date LaTeX diff after edits to the new Overleaf manuscript, compare the current `paper/main.tex` against the original submitted manuscript commit `d4b8ad5669f08a9107084211b05671bc286ce974` stored in the legacy manuscript Git history:

```bash
cd /Users/victorboussange/projects/MuScaRi
git --git-dir=.git/modules/paper \
  show d4b8ad5669f08a9107084211b05671bc286ce974:main.tex \
  > /tmp/muscari_previous_main_d4b8ad5.tex

cd paper
latexdiff --type=CFONT --disable-citation-markup \
  /tmp/muscari_previous_main_d4b8ad5.tex main.tex \
  > revision_diff_d4b8ad5.tex
```

Known `latexdiff` cleanup for this manuscript: if the generated diff introduces `\DIF...` markup inside the corresponding-author `\href` before the diff macros are defined, replace that author-email line in `revision_diff_d4b8ad5.tex` with the plain current line. If a deleted reference to the old Shapley figure leaves `fig:shapley` undefined, add a compatibility label next to the current scale-binned figure label in the diff source only, e.g. `\label{fig:shapley}\label{fig:scale_binned_rmse}`. Then compile from `paper/` with `latexmk -pdf -interaction=nonstopmode -halt-on-error revision_diff_d4b8ad5.tex`. Do not commit the generated diff source or PDF to the Overleaf `paper/` repo unless explicitly requested.

## Build, Test, and Development Commands

- `uv sync`: install the root project dependencies from `pyproject.toml`.
- `uv pip install torch --torch-backend=auto`: install a Torch build appropriate for the local machine.
- `uv pip install -e .`: install the root package in editable mode.
- `uv run python scripts/train.py`: run the main training entry point after data preparation.
- `./run_script.sh scripts/train.py optional_prefix`: launch a long-running script with logs in `stdout/`.
- `cd widget && uv sync && uv run python app.py`: run the local Gradio widget.
- `cd widget && uv run --extra dev pytest -m "not slow"`: run the default widget test suite.

## Coding Style & Naming Conventions

Use Python 3.11+ and follow the existing style: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and uppercase names for constants. Keep workflow scripts explicit; prefer small helpers in `muscari/` when logic is reused. No repository-wide formatter is configured, so keep imports tidy and match neighboring code.


## Writing style
You should stick to the style of the manuscript in `paper/`, following the style of high-impact journal's article. The current manuscript is under revision for publication in Nature Communications.

For the response to reviewers, when referring to the main text, use "LXXX" as a placeholder to refer to the lines of interest. Once the manuscript will be finalized, we will replace these placeholders with the actual line numbers.

## Testing Guidelines

The active pytest configuration is in `widget/pyproject.toml`; tests are named `test_*.py` under `widget/tests/`. Slow tests that download or load pretrained models and rasters are marked `slow`:

```bash
cd widget
MUSCARI_RUN_SLOW=1 uv run --extra dev pytest
```

For core package changes without existing tests, add focused tests near the affected subproject or document the manual verification command used.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase summaries such as `small fix` and `added modules`. Keep commits concise but descriptive, preferably imperative, for example `fix widget geometry bounds`. Pull requests should include purpose, commands run, data/model download implications, linked issues when relevant, and screenshots for widget UI changes.

## Security & Configuration Tips

Do not commit large generated datasets, model caches, secrets, or local credentials. The widget supports cache and model overrides through environment variables such as `MUSCARI_WIDGET_CACHE`, `MUSCARI_MODEL_PATH`, `MUSCARI_FEATURES_DIR`, and `MUSCARI_DEVICE`; prefer these over hard-coded local paths.
