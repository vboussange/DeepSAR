# Tightening the Benchmark Statistics: Minimal Defensible Compromise

## Decision

Use the cross-validation split as the inferential unit. For each model contrast, compare RMSE values from the same split with a two-sided paired $t$-test, and control the family-wise error rate with Holm adjustment. Treat the five split-level paired differences as independent replicate blocks for this analysis, while stating explicitly that this is a working approximation: held-out test folds are distinct, but training sets overlap across cross-validation runs and the same GIFT inventories are reused.

This is conditional inference for the five predefined spatial cross-validation splits. It is not evidence that cross-validation fits are strictly independent, and it must not be extended by treating generated spatial units, vegetation plots, or GIFT regions as independent replicates.

The 100 km spatial-block benchmark remains the sensitivity analysis for broader spatial dependence. Its five splits will be kept separate from the five 1 km splits rather than pooled into an artificial sample of ten.

## Why this is the minimal compromise

- Pairing is required because competing models are evaluated on the same fold. Independent-samples tests discard that blocking and answer the wrong comparison.
- Holm adjustment controls family-wise error like Bonferroni but is uniformly at least as powerful.
- Fold-level testing avoids pseudoreplication from overlapping generated spatial units and shared vegetation plots.
- Explicitly declaring the working independence assumption is more accurate than calling the trained models or cross-validation splits themselves independent.
- Reporting the paired effect and its confidence interval keeps the practical magnitude visible despite the very small inferential sample ($n=5$, $df=4$).

## Scope

This sprint changes only the benchmark-comparison statistics, their plotted compact-letter display, and the minimum manuscript/rebuttal wording needed to describe them accurately. It does not retrain models, alter data splits, change performance metrics, revise the SGD motivation in `paper/main.tex`, or introduce a mixed model, hierarchical bootstrap, repeated cross-validation, or a new benchmark.

## Implementation plan

### 1. Freeze the comparison families and inputs

- Use the stored fold-level columns `interp_rmse` and `extrap_rmse` from the benchmark result files.
- Audit Figure 3's source, rendered panels, caption, and manuscript model list before fixing family sizes. The current plotting `label_map` omits the log-linear baseline although the manuscript caption mentions it; resolve that source/caption mismatch explicitly rather than silently testing a different model set.
- Define one multiplicity family per Figure 3 performance panel:
  - interpolation: all pairwise comparisons among models actually displayed and having interpolation RMSE values;
  - extrapolation: all pairwise comparisons among models actually displayed and having extrapolation RMSE values.
- Do not combine interpolation and extrapolation comparisons into one family. Do not add the 100 km results to either 1 km family.

### 2. Replace the current testing code with one paired implementation

Modify `figures/figure_3/figure_3.py` so that the statistical report and compact-letter display use the same result object.

- Pivot each endpoint to a fold-by-model table using `fold` as the row key.
- Validate that every `(experiment, fold)` combination is unique and that each tested pair has the expected five matching finite fold values. Fail loudly on duplicates, missing folds, or an unmatched model instead of silently comparing different arrays.
- For every model pair, calculate the fold-wise difference in RMSE and run a two-sided paired $t$-test (equivalently, a one-sample $t$-test of the five differences against zero).
- Record, with an explicit and consistent sign convention:
  - both model names;
  - endpoint and fold IDs;
  - $n$ and $df$;
  - mean paired RMSE difference;
  - percentage difference relative to the named reference model;
  - 95% $t$ confidence interval for the mean paired difference;
  - $t$ statistic;
  - raw $P$ value;
  - Holm-adjusted $P$ value;
  - rejection decision at $\alpha=0.05$.
- Apply `statsmodels.stats.multitest.multipletests(..., method="holm")` separately to the complete interpolation and extrapolation families.
- Build the compact-letter matrix from the Holm-adjusted $P$ values. Remove the current `ttest_ind`/`MultiComparison.allpairtest` route, which treats matched folds as independent groups and currently feeds unadjusted $P$ values to the letters.
- Make `report_model_performance` print or save the same paired results used for the letters, so there is no second statistical implementation that can drift.

Keep `muscari/cld.py` unchanged unless the shared code genuinely requires modification. Its generic parser is used elsewhere; bypassing it for a purpose-built adjusted-$P$ matrix is the smaller and safer change.

### 3. Add focused verification for the helper

Add small automated tests or executable assertions covering:

- invariance to input row and fold order;
- correct matching by fold rather than array position;
- rejection of duplicated or missing model-fold values;
- agreement with `scipy.stats.ttest_rel` on a hand-checkable example;
- correct Holm-adjusted values and family membership;
- agreement between the adjusted matrix, rejection decisions, and compact-letter display.

With only five pairs, normality diagnostics have little power and should not be used to select a more favorable test. Retain the individual fold points in Figure 3 so that sign reversals or a result driven by one fold remain visible.

### 4. Regenerate and audit the statistical outputs

- Run the Figure 3 pipeline from the full benchmark result files, not the smoke-test CSV.
- Save an auditable pairwise-results table alongside the Figure 3 outputs or benchmark results.
- Confirm that the focal environment-only versus environment-plus-area result in the Results is reproduced from that table; do not preserve the current hard-coded $t_4$ or adjusted $P$ value unless it matches.
- Check every letter in both panels against the Holm-adjusted comparison matrix.
- Check that all directional claims are supported by the fold-wise differences, not just the ordering of aggregate means.
- Compare the direction and magnitude of the focal 1 km conclusions with the existing 100 km benchmark. Do not claim robustness where the direction changes; describe such a result as sensitivity to spatial blocking.

### 5. Make only the necessary manuscript changes

In `paper/main.tex`:

- Add a short Methods statement defining the split-level paired analysis, the two panel-wise Holm families, $\alpha=0.05$, and the working independence approximation.
- Remove or neutralize wording that calls the five trained models or cross-validation splits "independent." Use "five spatial-block cross-validation splits" instead. Independence should appear only as the stated assumption applied to the five paired split-level differences.
- Keep the Figure 3 caption's paired/Holm description, after verifying that it matches the regenerated code and model membership.
- Update the focal Results contrast from generated output and include the mean paired RMSE difference and 95% confidence interval if it fits without restructuring the paragraph.
- For an adjusted $P>0.05$, write "we found no clear evidence of a difference" rather than treating non-significance as evidence of equivalence.
- Preserve the existing 100 km analysis as a separate spatial robustness check.
- Do not alter the end-to-end stochastic-gradient-descent justification around line 153; the benchmark inference does not test or replace that methodological rationale.

In `paper/response_to_reviewers_3rd_rev.md`, change only wording made inaccurate by this sprint, especially any claim that the five cross-validation models or splits are themselves independent. Add a concise description of the paired fold-level comparison only if needed to keep the rebuttal consistent with the Methods.

### 6. Final checks

- Re-run the relevant Python tests and Figure 3 generation.
- Verify the focal statistic, confidence interval, adjusted $P$ value, and compact letters directly against the saved pairwise-results table.
- Compile `paper/main.tex` with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from the `paper` directory.
- Inspect the PDF to confirm that the Figure 3 panel membership, letters, caption, Results, Methods, and supplementary performance tables agree.
- Review the final diffs in both the root repository and the nested `paper` repository, preserving all unrelated uncommitted manuscript edits.

## Acceptance criteria

- No independent-samples or unadjusted pairwise test determines Figure 3 letters or manuscript significance claims.
- Every tested contrast uses exactly matched fold IDs and reports $n=5$ and $df=4$ when all folds are present.
- Holm adjustment is applied once within each documented panel-wise family.
- The report, saved results, figure letters, caption, Results, and Methods all describe the same analysis.
- The manuscript presents independence as a transparent working assumption and does not imply that overlapping cross-validation fits are literally independent.
- The 1 km and 100 km benchmarks are not pooled, and any lack of agreement is reported rather than hidden.
- The manuscript compiles without new LaTeX, citation, or reference errors.

## Deliberate limitations

This plan does not remove all dependence induced by overlapping training sets. A corrected resampled test, repeated spatial cross-validation, or hierarchical resampling would address that issue more directly but would exceed the requested minimal-change compromise or introduce additional assumptions. The paired-Holm analysis should therefore be presented as approximate, split-conditional formal inference supported by effect estimates, confidence intervals, visible fold-level results, and the separate 100 km sensitivity benchmark.
