# Third-Revision Results-Section Audit Plan

## Objective

Address the actionable findings from the independent audit of
`paper/main.tex` lines 226--253 and resolve the statistical-significance TODO
at line 186 while preserving the intended presentation and narrative of the
third-revision Results section. Changes should remain local to the 100 km
benchmark comparison, Figure 5, their supporting code and outputs, and the
associated manuscript text.

## Decisions already made

| Audit finding | Decision | Planned response |
|---|---|---|
| Display-only 2-by-2 smoothing and finite-difference estimates of $dS_T/dA$ | Keep the current presentation. | Do not discuss display smoothing in the manuscript. Retain the interpretation of the finite differences as estimates of the derivative and the current concise caption wording. |
| Map and location-specific SAR pipelines differ | Investigate regenerating the maps if exact harmonization is low effort. | Run a focused feasibility check, then either regenerate all Figure 5 map products with the location-SAR post-processing or retain the present maps and restrict the prose to claims shared by both pipelines. |
| Site-level model agreement is uneven | Address briefly in Results. | Add one sentence reporting that the endpoint increase is positive in 5/5 ensemble members for Parc Naziunal Svizzer, 3/5 for Žďárské vrchy, and 2/5 for Nationaal Park Veluwezoom. Describe the curves explicitly as ensemble-mean patterns. |
| Close agreement with the Czech flora map | Keep the comparison. | Preserve the current close-agreement statement and citation to `klimova2025`. |
| Ecological interpretation may be too strong | Keep the paragraph structure but use more cautious wording. | Separate model output from ecological interpretation and use “consistent with” language without attributing causal effects to topography or habitat heterogeneity. |
| Scale-binned performance concerns finite-effort interpolation rather than asymptotic $S_T$ | Keep the comparison. | Replace “conferring” with precise wording that the analysis *showed better finite-effort interpolation performance* at coarser grains, without presenting it as direct validation of mapped $S_T$. |
| The 100 km GIFT comparison lacks fold-paired statistics | Resolve the TODO using the existing manuscript convention. | Test the environment-and-area versus environment-only contrast across matched folds, report the mean paired difference, 95% CI, and Holm-adjusted $P$ value for NRMSE, and assess bias calibration using absolute fold-level median relative bias. |

The following audit suggestions are therefore out of scope unless the
regeneration step exposes a substantive error: documenting the display-only
smoothing, replacing the finite-difference derivative terminology, removing the
Klímová comparison, and removing the scale-binned performance comparison.

## Implementation plan

### 1. Record the task and freeze the current reference outputs

- [ ] Add a `running` entry to `agent_log.yaml` before changing or regenerating
  Figure 5 products. Record the selected environment-only ensemble
  (`dae0789a3c87`), dataset ID (`ceacce0`), published model revision, current
  Figure 5 checksums, and the decisions above.
- [ ] Reproduce the current focal-location values and per-member endpoint
  increases from the selected five-member ensemble.
- [ ] Save a compact diagnostic table outside the manuscript containing direct
  predictions, memberwise monotone-adjusted predictions, and the number of
  members with a positive 25-to-2,500 km$^2$ increase at each focal location.

### 2. Test whether map/SAR harmonization is genuinely low effort

- [ ] Build a smoke test for the three marked locations and one small spatial
  tile using the same centered nested-window geometry, memberwise cumulative
  maximum, and ensemble aggregation used by `calculate_sar.py`.
- [ ] Compare these values with the current direct map predictions at 5 and
  50 km and with the values used in panel e and Table S7.
- [ ] Estimate the runtime and memory required to apply this calculation across
  Europe without introducing a general projection framework or broad refactor.

Decision gate:

- **Regenerate** if the exact, centered, memberwise monotone-adjusted map can be
  produced with a small local modification and a single tractable projection
  job. Apply the same treatment to $S_T$, between-model standard deviation, and
  the paired predictions used for $dS_T/dA$ at both displayed grains.
- **Retain the current maps** if exact harmonization requires projecting the full
  100-scale SAR grid across Europe or otherwise ceases to be a low-effort local
  change. In that case, make no processing disclosure requested under audit
  finding 1, but ensure the Results text does not imply numerical identity
  between map pixels and the monotone-adjusted location-specific curves.

### 3. Regenerate Figure 5 products if the decision gate passes

- [ ] Make the smallest possible change in
  `figures/figure_5/calculate_SR_dSR.py`; keep all resolutions and model paths as
  visible top-level constants and preserve existing output names where valid.
- [ ] Run the full projection through `run_script.sh` under the `uv`
  environment, then regenerate the location SAR data, Table S7, Figure 5, and
  supplementary dispersion maps.
- [ ] Confirm that the A1/A2, B1/B2, and C1/C2 values use the intended common
  estimand after regeneration. If numerical values change, update Table S7 and
  all corresponding prose together.
- [ ] Retain the existing display smoothing in `figure_5_panels.py`.

### 4. Tighten the Results prose while preserving its structure

- [ ] Keep the paragraph order: mapping setup; richness maps; accumulation-rate
  maps; illustrative SARs; interpretation caveats.
- [ ] Revise the cross-grain statement on line 233 so that the 5 km map
  “resolves finer within-region variation” rather than claiming to locate where
  the species of a 50 km unit are concentrated.
- [ ] Qualify the site descriptions as ensemble-mean patterns. Prefer
  “smallest ensemble-mean increase” for Veluwezoom and “largest ensemble-mean
  increase” for Parc Naziunal Svizzer while retaining the existing numerical
  values and curve descriptions where supported.
- [ ] Add one compact sentence reporting per-member agreement:
  `5/5` for Parc Naziunal Svizzer, `3/5` for Žďárské vrchy, and `2/5` for
  Nationaal Park Veluwezoom.
- [ ] Preserve the statement that the 5 km Žďárské vrchy prediction is in close
  agreement with the external Czech flora map and retain `\citep{klimova2025}`.
- [ ] Retain the Swiss landscape interpretation, but state that the predicted
  pattern is *consistent with* topographic and habitat heterogeneity rather than
  caused by them. Keep the current paragraph structure.
- [ ] Rewrite the final caveat sentence to state that the scale-binned analysis
  showed better **finite-effort interpolation** performance at coarser grains.
  Retain the links to `fig:scale_binned_rmse`, `figSI:maps_std`, and
  `figSI:EVA_locations`.
- [ ] Use “between-model standard deviation” or “between-model dispersion” when
  referring to the five cross-validation models; do not call it a calibrated
  confidence or prediction interval.

### 5. Keep the caption and Methods synchronized with any regenerated output

- [ ] If maps are regenerated, update only the processing statements that become
  factually necessary; retain the concise description of panels c--d as
  finite-difference estimates of $dS_T/dA$.
- [ ] Ensure Methods, Figure 5 caption, Table S7 caption, and Results use the
  same ensemble aggregation, spatial-unit geometry, target areas, and
  uncertainty terminology.
- [ ] If the nearest log-spaced areas remain 26.298 and 2,414.443 km$^2$, keep
  the current rounded 25 and 2,500 km$^2$ presentation consistently throughout.

### 6. Resolve the 100 km GIFT significance TODO on line 186

- [ ] Use the `asymptotic_total`, `fold_member` rows in
  `scripts/results/gift_asymptote_evaluation/d0848f6/gift_asymptote_evaluation_results.csv`
  and verify that the environment-only and environment-and-area models are
  compared on the same five folds and GIFT samples.
- [ ] Reproduce the aggregate values currently reported in the manuscript:
  38.02% NRMSE for environment only and 32.90% for environment and area, plus
  the corresponding fold-level median relative-bias values.
- [ ] For the primary NRMSE contrast, calculate fold-paired differences as
  `environment-and-area minus environment-only`, their mean, a two-sided 95%
  $t$ confidence interval, and a two-sided paired $t$-test. Apply Holm
  adjustment over all pairwise comparisons among the three 100 km MuScaRi
  feature variants, matching the Figure 3 statistical convention.
- [ ] Define fold-level bias-calibration error as
  `abs(median_relative_bias)` because calibration improves as relative bias
  approaches zero. Analyze paired differences in this quantity using the same
  confidence-interval and Holm-adjustment procedure, with a separate
  metric-specific comparison family.
- [ ] Because there are only five folds, calculate the exact paired sign-flip
  permutation $P$ value for both targeted contrasts as a sensitivity check.
  Use the paired $t$-test in the manuscript for consistency, but flag any
  qualitative disagreement between the tests.
- [ ] Write the fold-level values and pairwise statistics to a compact CSV and
  Markdown summary under
  `scripts/results/gift_asymptote_evaluation/d0848f6/`. Prefer a small local
  analysis script in `scripts/experiments_3rd_rev/` over embedding unrecorded
  calculations in the manuscript workflow.
- [ ] Update line 186 according to the evidence:
  - if the adjusted test supports a difference, report the paired effect, 95%
    CI, and Holm-adjusted $P$ value;
  - otherwise state that the combined model had the lower aggregate NRMSE but
    that there was no clear evidence of a fold-level difference;
  - retain “better bias calibration” only if the absolute-bias analysis
    supports it, otherwise describe only the aggregate bias values or remove
    the comparative claim.
- [ ] Remove the inline TODO and ensure the statistical wording matches the
  reporting style already used for the 1 km interpolation and GIFT comparisons
  on lines 179 and 184.

## Verification

- [ ] Run all Python scripts with `uv` and compile modified scripts with
  `uv run python -m py_compile`.
- [ ] Numerically verify finite and non-negative $S_T$ and $dS_T/dA$ rasters,
  raster CRS/alignment, expected dimensions, and absence of decreasing steps in
  all displayed monotone-adjusted SAR members.
- [ ] Verify the focal-location means, standard deviations, endpoint
  differences, and member-agreement counts against generated artifacts.
- [ ] Verify the 100 km fold pairing, aggregate NRMSE and relative-bias values,
  confidence intervals, raw and Holm-adjusted paired-test results, and exact
  permutation sensitivity results from a clean script run.
- [ ] Visually inspect the regenerated Figure 5 and supplementary dispersion
  maps at full resolution.
- [ ] Run `uvx codespell`, a citation/label/reference audit, `git diff --check`,
  and a full `uv run latexmk -pdf -interaction=nonstopmode -halt-on-error
  main.tex` build from `paper/`.
- [ ] Confirm that no unrelated dirty files or generated smoke-test products are
  staged.

## Completion and pause protocol

- [ ] Update `agent_log.yaml` with the decision at the map-regeneration gate,
  executed commands, output paths, numerical findings, warnings, and final
  outcome.
- [ ] Report the exact manuscript/code/figure changes and checks to the user.
- [ ] Pause for user verification before committing. Keep any eventual commit
  narrow and exclude pre-existing unrelated changes.

## Expected files in scope

- `paper/main.tex`
- `scripts/experiments_3rd_rev/gift_100km_pairwise_statistics.py` (or an
  equivalently small existing analysis script extended locally)
- `scripts/results/gift_asymptote_evaluation/d0848f6/` pairwise-statistics CSV
  and Markdown summary
- `figures/figure_5/calculate_SR_dSR.py` (conditional on regeneration)
- `figures/figure_5/calculate_sar.py` (only if exact target areas or diagnostics
  require a local adjustment)
- `figures/figure_5/figure_5_panels.py` (only if regenerated artifacts require a
  local loader change; retain display smoothing)
- `figures/figure_5/SARs/SARs_table.tex` and generated Figure 5 artifacts
- `figures/SI/projection_errors/figure_std.py` and generated supplementary map
  artifacts, if map products change
- `paper/figures/figure_5.*` and affected supplementary figures
- `agent_log.yaml`
