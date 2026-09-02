# Pre-filtered Majority Voting versus Standard Stacking (Spambase)

Compares four decision-fusion strategies on the UCI Spambase dataset, using one shared
train/validation/test split and one shared set of six heterogeneous base learners:

1. **Standalone XGBoost** — single-model baseline.
2. **Hard voting** — majority class across the six base learners; 3:3 ties are broken by the
   average signed distance from each learner's own threshold.
3. **Standard stacking** — an XGBoost meta-learner trained on out-of-fold base-learner
   probabilities.
4. **Pre-filtering (paper-inspired)** — an instance bypasses the meta-learner entirely when at
   least `n - 1` of the `n` base learners agree (5 of 6 here); its vote is accepted directly. Only
   the remaining "difficult" instances are routed to an XGBoost meta-learner trained specifically
   on that filtered subset.

The pre-filtering rule is adapted from a paper that used seven homogeneous DNNs with a
delete-one-seventh bootstrap. Here the `n-1`-agreement fusion rule is isolated and tested against
six existing heterogeneous base learners; the DNN architecture and the bootstrap construction are
not reproduced.

**Headline result, over 70 split seeds: pre-filtering does not beat standard stacking on this
dataset.** It is slightly but consistently *worse* (mean −0.0012 balanced accuracy, losing on 42 of
70 seeds), while beating standalone XGBoost and plain hard voting by small margins. Full numbers,
including the paired tests, are in [Results](#results) below.

## Contents

| Path | Description |
|---|---|
| `prefilter_majority_voting_comparison_spambase.Rmd` | The complete analysis — the version to run locally. |
| `spambase.data` | UCI Spambase: 4601 rows x 58 columns, CSV, no header row. |
| `spambase.names` | Attribute list defining the exact column order of `spambase.data`. |
| `spambase.DOCUMENTATION` | UCI dataset writeup: source, collection method, per-attribute statistics. |
| `SpambasePrefilter5of6OOFThresholds_70seed_summary.csv` | Per-seed metrics for the 70-seed sweep — the source of every number in [Results](#results). |
| `prefilter_seed_results/` | Committed output of one reference run (`split_seed = 700`). |
| `hpc/` | The cluster scripts that produced the 70-seed sweep — see [Results](#results). |

## Requirements

R (developed on 4.4.2) plus:

```r
install.packages(c(
  "rmarkdown", "knitr",
  "caret", "xgboost", "glmnet", "ranger", "mgcv", "e1071", "earth",
  "dplyr", "tidyr", "ggplot2", "scales"
))
```

`scales` and `knitr` are called via `::` rather than `library()`, but must be installed.

## Running it

The document is parameterised. From the repository root:

```r
rmarkdown::render(
  "prefilter_majority_voting_comparison_spambase.Rmd",
  params = list(
    split_seed = 700,
    configuration_id = "SpambasePrefilter5of6OOFThresholds"
  )
)
```

or open it in RStudio and use **Knit** (which uses the defaults in the YAML header: seed `700`,
configuration `SpambasePrefilter5of6OOFThresholds`).

`spambase.data` is read by relative path, so the working directory must be the repository root —
which is what `rmarkdown::render()` and RStudio's Knit both do by default.

Expect a long run. The tuned radial SVM grid search and the GAM are refit inside every one of the
five out-of-fold folds, which dominates the runtime; on a desktop this is hours, not minutes, per
seed.

### Repeating across seeds

`configuration_id` names the output files; `split_seed` selects the split. Re-rendering with a new
seed under the same `configuration_id` **appends** a row to the summary CSV, and re-rendering with a
seed already present **replaces** that seed's rows. The cross-seed tables and plots only become
informative once several seeds have accumulated — correlations across seeds return `NA` until at
least three are present.

## Outputs

Written to `prefilter_seed_results/`, keyed by `configuration_id`:

- `<configuration_id>.csv` — one row per seed: balanced accuracy for all four strategies, the
  pre-filter deltas, routing rates, and every base learner's threshold and test metrics.
- `<configuration_id>_test_predictions.csv` — one row per test observation: base-learner
  probabilities, per-learner votes, which route the instance took, and each strategy's prediction
  and correctness.
- `<configuration_id>_pairwise_diversity.csv` — all 15 base-learner pairs, with disagreement rate,
  double-fault rate and the Q statistic.
- `plots/<configuration_id>/` — cross-seed comparison plots, plus `diagnostics/` for base-learner
  distributions, pairwise-diversity heatmaps and per-route performance.

## Results

Evaluated over **70 split seeds** (100 to 7000 in steps of 100), each producing an independent
64/16/20 split and a 919-row test set. Every figure below is computed from the observation-level
predictions of that sweep — 64,330 rows, one per test instance per seed — recomputing balanced
accuracy per seed from `Actual_Class` and each strategy's prediction column.

The seeds were run on the Hábrók HPC cluster rather than locally — the analysis is the same, but
70 sequential renders at hours apiece is not something you want to sit through. The results are
committed here so you don't have to repeat them.

The scripts that produced the sweep are in `hpc/`: `prefilter_majority_voting_comparison_spambase_habrok.R`,
a plain `Rscript` port of the `.Rmd`, and `submit_habrok.sh`, the SLURM array job that ran it once per
seed. They are committed for transparency about what was actually run, not for reuse — the data path
and the project directory are hardcoded to the cluster account, so they will not run elsewhere
unedited. Most people do not have Hábrók access; the `.Rmd` at the repository root is the version
to run locally.

Everything that differs is infrastructural: command-line arguments instead of knitr params, a fixed
cluster data path, out-of-fold folds run in parallel with thread pinning, file-locked shared CSVs,
console output instead of knitted tables. The split, the base learners and their hyperparameters,
the per-model-per-fold seeds, the fusion rules, the threshold sweeps and the output schema are
identical. See [On exact reproducibility](#on-exact-reproducibility) for what the change of
environment does cost.

Those per-seed figures are committed as
`SpambasePrefilter5of6OOFThresholds_70seed_summary.csv` (70 rows — one per seed, with each
strategy's balanced accuracy and accuracy, the routing counts, per-route accuracy, and the paired
differences), so every table and test below can be recomputed from this repository. The
observation-level file it was derived from is ~18 MB and is not included.

Balanced accuracy on the test set, aggregated across seeds:

| Strategy | Mean BA | SD | Min | Max | Best on |
|---|---|---|---|---|---|
| **Standard stack** | **0.9462** | 0.0076 | 0.9249 | 0.9631 | 32/70 seeds |
| Pre-filtered stack | 0.9451 | 0.0077 | 0.9265 | 0.9595 | 19/70 seeds |
| Hard voting | 0.9435 | 0.0080 | 0.9176 | 0.9572 | 11/70 seeds |
| Standalone XGBoost | 0.9396 | 0.0120 | 0.9004 | 0.9600 | 8/70 seeds |

Because every strategy sees the identical split within a seed, the honest comparison is the paired
difference. Pre-filtered stack minus each alternative, across the 70 seeds:

| Comparison | Mean difference | 95% CI | Win / tie / loss | Paired t | Wilcoxon |
|---|---|---|---|---|---|
| vs standard stack | **−0.0012** | [−0.0022, −0.0001] | 27 / 1 / 42 | p = 0.035 | p = 0.041 |
| vs hard voting | +0.0015 | [+0.0006, +0.0025] | 47 / 2 / 21 | p = 0.0021 | p = 0.0012 |
| vs standalone XGBoost | +0.0055 | [+0.0034, +0.0075] | 53 / 0 / 17 | p = 1.2e-06 | p = 3.6e-07 |

Read plainly:

- **The pre-filter loses to standard stacking.** The deficit is small but statistically detectable,
  and it loses on 42 of 70 seeds. Routing the confident cases around the meta-learner costs
  something rather than buying something.
- **It does beat hard voting and standalone XGBoost**, so the meta-learner still earns its place on
  the difficult cases — the gate is what hurts, not the stacking.
- **Every effect is small next to the noise.** Seed-to-seed SD is roughly 0.008, while the largest
  paired effect is 0.0055. Any single seed can show the opposite ordering, so no conclusion here
  should be drawn from one split.

### Routing behaviour

The 5-of-6 agreement gate accepts a direct vote for **94.9%** of test instances on average
(SD 0.8pp, range 92.8–96.8%), leaving a mean of **47 of 919** cases per seed for the meta-learner.

Pooled over all 70 seeds, split by the route an instance took:

| Route | Instances | Accuracy | Balanced accuracy |
|---|---|---|---|
| Direct vote | 61,060 | 0.9618 | 0.9602 |
| Meta-learner | 3,270 | 0.6642 | 0.6603 |

The gate does isolate genuinely hard cases — accuracy on the routed 5% is close to a coin flip.
But that subset is small and its class balance is similar to the whole (37% vs 39% spam), so the
meta-learner is being retrained on a few hundred difficult training rows, which is the most likely
source of the pre-filter's instability.

### Committed reference run

`prefilter_seed_results/` contains **one** run, at `split_seed = 700`, produced by this `.Rmd` on a
desktop. It is not representative: seed 700 is one of the 27 seeds where the pre-filter happens to
beat the standard stack. Its cross-seed plots also degenerate to a single point each. Treat it as a
worked example of the output format, not as evidence — the table above is the evidence.

### On exact reproducibility

Everything is seeded (`set.seed(500 + fold)`-style offsets, preserved per model per fold), so a
given seed is deterministic within one environment. Across environments it is not quite: XGBoost,
ranger and the underlying solvers differ in their last decimals with thread count, BLAS build and
package version. The vote gate and the threshold sweeps are discrete, so those tiny shifts can flip
a vote or move a selected threshold, and the pre-filtered meta-learner — trained on the smallest
subset — is the most sensitive of the four strategies. Seed 700 illustrates it: the committed
desktop run and the cluster sweep agree exactly on standalone XGBoost and hard voting, and differ
by 0.001 and 0.006 balanced accuracy on the standard and pre-filtered stacks. Expect to reproduce
the ordering and the magnitudes, not the digits.

## Method notes

- **Split**: 64/16/20 train/validation/test via `caret::createDataPartition`, stratified on the
  class label.
- **Base learners**: elastic-net logistic regression (`glmnet`), GAM (`mgcv`), random forest
  (`ranger`), tuned radial SVM (`e1071::tune.svm`), MARS (`earth`), XGBoost. Class imbalance is
  handled per model (`scale_pos_weight`, `class.weights`).
- **Meta-features**: 5-fold CV regenerates every base learner per fold to produce leakage-free
  out-of-fold predictions for the meta-learner's training input. Validation and test meta-features
  instead reuse the base models fitted on the full training set.
- **Thresholds**: the voting gate uses one threshold per learner, chosen on the out-of-fold
  training predictions to maximise balanced accuracy. Model-selection thresholds for standalone
  XGBoost and both meta-learners are swept over 0.05–0.95 on the validation set, also on balanced
  accuracy — the dataset is imbalanced, so 0.5 is not assumed.

## Data

Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. *Spambase* [Dataset]. UCI Machine Learning
Repository, 1999. <https://doi.org/10.24432/C53G6X>

Licensed CC BY 4.0. The class label is the final column (`spam`: 1 = spam, 0 = non-spam).
