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

## Contents

| Path | Description |
|---|---|
| `prefilter_majority_voting_comparison_spambase.Rmd` | The complete analysis — the only code file. |
| `spambase.data` | UCI Spambase: 4601 rows x 58 columns, CSV, no header row. |
| `spambase.names` | Attribute list defining the exact column order of `spambase.data`. |
| `spambase.DOCUMENTATION` | UCI dataset writeup: source, collection method, per-attribute statistics. |
| `prefilter_seed_results/` | Committed output of one reference run (`split_seed = 700`). |

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

## Reference run

`prefilter_seed_results/` holds one committed run at `split_seed = 700`. Test-set balanced accuracy:

| Strategy | Balanced accuracy |
|---|---|
| Standalone XGBoost | 0.9414 |
| Hard voting | 0.9545 |
| Standard stack | 0.9544 |
| Pre-filtered stack | 0.9558 |

93.9% of test instances cleared the 5-of-6 agreement gate and bypassed the meta-learner; 56 were
routed to it. Note that the plots in this directory are drawn from a single seed, so the
across-seed views degenerate to one point each.

Reproducing this exactly requires the same package versions — the base learners are individually
seeded (`set.seed(500 + fold)`-style offsets, preserved per model per fold), so the split and model
fits are deterministic, but XGBoost and ranger can differ in the last few decimal places across
thread counts and library versions.

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
