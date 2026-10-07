# Explain fitted models

The engine offers **permutation importance** and **Tree SHAP** without retraining. Supply fitted scalar regression models and their prepared feature columns. These tools work on your selected evaluation population, including a sector, issuer, size group, maturity group, or any custom filter.

Comparison-bound results use the applied comparison's error scale and unit. They also include
`comparison_source_position` (a position in the applied `comparison.data`) and, when mapped, `record_id`.
`row_position` remains relative to the selected explanation cohort. This distinction preserves traceability
when paired coverage excludes rows or the DataFrame has duplicate index labels. Use a unique Record ID
for durable linkage across separately filtered reviews. Direct explanation functions do not add comparison metadata.

| Question | Tool | What a large value means |
|---|---|---|
| Which inputs does this fitted model rely on for held-out accuracy? | Permutation importance | Shuffling the input increased the final-target error. |
| What pushed a particular prediction higher or lower? | Local Tree SHAP | That feature contributed strongly to the learned prediction relative to the tree expectation. |
| Which features most influence predictions in a selected population? | Global Tree SHAP | Mean absolute contribution is large; this does **not** establish better accuracy. |
| Does a related group of features matter together? | Grouped permutation | Shuffling that group jointly increased error while retaining relationships within the group. |
| Does a feature matter beyond a supplied context? | Within-context permutation | Shuffling only among records in the same context increased error. |

Use held-out data appropriate for the question. The engine cannot infer whether a dataframe was used for training. Repeatedly using a locked test set to choose features compromises its role as a final independent check.

## Start with fitted models

This example assumes you already have a prepared `evaluation_data` dataframe, three fitted models, and their ordered feature lists. Replace the mapped column names with yours. Models may use different features. Preserve preprocessing, pandas categorical types, category levels, and feature order from training.

```python
# SETUP LOGIC: Import the registry and interactive review API.
from model_comparison_engine import Model, compare_model_set, show_comparison

# CONFIGURATION LOGIC: The residual model predicts actual_spread minus anchor_spread.
models = [
    Model("Baseline", baseline_model, baseline_features),
    Model("Expanded", expanded_model, expanded_features),
    Model("Residual", residual_model, residual_features, offset="anchor_spread"),
]

# INFERENCE LOGIC: Predict once per fitted model and initially select this pair.
review = compare_model_set(
    evaluation_data, actual="actual_spread", models=models,
    reference="Baseline", candidate="Residual",
    time_column="event_time", entity_column="instrument_id", unit="bps",
)

# UI LOGIC: Change the selected reference/candidate pair in the review controls.
panel = show_comparison(review)
```

`compare_models()` still accepts exactly two fitted `Model` objects. `compare_prediction_set()` accepts any number of **saved prediction columns**, but prediction-only inputs cannot support model explanations: SHAP needs model internals, and permutation needs repeated inference on modified feature inputs.

`review.select_models("Expanded", "Residual")` returns a comparison of those two cached predictions without calling a model again. Keep the returned object if you want to use that pair in later API calls.

Models remain in memory. Reports contain explanation tables, settings, and figures; the engine does not write or load pickle/joblib model artifacts. Load only your own trusted model artifacts through their appropriate library before passing a fitted object.

## Permutation importance

```python
# EXPLANATION LOGIC: Use the current selected pair's common finite evaluation cohort.
importance = review.permutation_importance(
    model="Residual", population="paired", metric="mae",
    max_rows=2000, repeats=5, random_state=7,
)

# PLOTTING LOGIC: Show loss increase and shuffle variability without printing a large table.
from model_comparison_engine.explanation_plots import plot_permutation
display(plot_permutation(importance, top_n=20))
```

The default is a reproducible sample of at most 2,000 eligible records. Each repeat shuffles one feature, predicts again, reconstructs the final target, and compares error against the **same fixed baseline rows**. MAE and RMSE are both calculated; `metric` chooses the importance ranking and plotted loss. No rows are dropped because a shuffled prediction became invalid: the request fails with an explanation instead.

Read the returned object as follows:

- `importance.tables["importance"]`: mean and standard deviation of the MAE/RMSE increase by feature or feature group.
- `importance.tables["baseline"]`: original sampled MAE, RMSE, and evaluated count.
- `importance.tables["repeats"]`: every shuffle's losses and changes.
- `importance.tables["rows"]`: exact positional record identifiers and original pandas index labels.
- `importance.settings`: input, eligible, sampled and evaluated counts, invalid baseline predictions, seed, offset and shuffle context.
- `importance.notes`: interpretation and limitations.

Positive loss increase indicates reliance. Negative importance is retained: shuffling sometimes helps, especially for noisy or harmful inputs. Repeat standard deviation measures **random-shuffle variability**, not a confidence interval for generalization and not a significance test. The slice comparison's p-value correction controls do not turn permutation bars into significance tests.

### Related feature groups

```python
# CONFIGURATION LOGIC: Each named group contains actual feature columns used by this model.
feature_groups = {
    "Quote state": ["quote_mid", "quote_width", "quote_age"],
    "Quote movement": ["quote_change", "dealer_agreement"],
}

# EXPLANATION LOGIC: Compare two whole groups; omit separate one-column runs for speed.
group_importance = review.permutation_importance(
    model="Residual", groups=feature_groups, include_individual=False,
    max_rows=2000, repeats=5, random_state=7,
)
```

Group members use the **same row permutation**: for original pairs `[(1,10),(2,20),(3,30)]`, a permutation `[2,0,1]` produces `[(3,30),(1,10),(2,20)]`. It does not independently scramble each group member. This preserves internal relationships while breaking the group's association with the remaining inputs and actual outcome.

Related features can substitute for one another, making individual importance misleadingly small. Grouped permutation addresses this particular substitution problem; it still does not prove causal value or estimate the performance of retraining without those features.

### Shuffle within context

```python
# EXPLANATION LOGIC: Exchange features only within the supplied session and sector categories.
conditional_importance = review.permutation_importance(
    model="Residual", groups=feature_groups, include_individual=False,
    shuffle_within=["session_date", "sector"], max_rows=4000, repeats=5,
)
```

`session_date` must already exist if you choose it. The engine does not silently choose a time aggregation. Context groups are formed **after sampling**; small samples may leave singleton groups whose records cannot move. The result reports `movable_records` and rejects an all-singleton population. Missing context values form their own group.

This answers a different question from an unrestricted shuffle: predictive reliance **within the declared context**. It is not a general conditional-distribution estimator and does not make a causal claim. Neither method guarantees that permuted feature combinations are economically plausible.

### Cost, progress, and cancellation

Prediction calls equal `1 + number_of_groups × repeats`; the first call establishes the baseline. The default call budget is 500. Reduce selected groups/repeats or explicitly raise `max_predictions` when a larger run is intentional. Hard bounds are 50,000 sampled rows, 100 repeats, and 10,000 prediction calls. The original dataframe is never modified.

Pass `progress=callback` to receive a dictionary with `stage`, `completed`, and `total`. Raising an exception from that callback cancels before the next batch/repeat; an already running estimator prediction cannot be interrupted by this callback. The notebook UI updates progress on explicit explanation actions and does not rerun them when you only change display controls.

## Tree SHAP

```python
# EXPLANATION LOGIC: Explain sampled predictions and retain the largest sampled errors locally.
explanation = review.shap_values(
    model="Residual", population="paired", max_rows=500,
    max_local_rows=20, random_state=7,
)

# PLOTTING LOGIC: Inspect global magnitude/direction and the first saved local case.
from model_comparison_engine.explanation_plots import plot_shap_global, plot_shap_local
display(plot_shap_global(explanation))
display(plot_shap_local(explanation))
```

Fitted LightGBM regressors and native LightGBM Boosters use their native `pred_contrib=True` TreeSHAP implementation. This needs no separate SHAP installation and follows the model's prediction iteration defaults, including its selected best iteration. Other supported fitted regression trees use the optional `shap.TreeExplainer` package:

```bash
pip install 'model-comparison-engine[explain]'
```

The SHAP backend uses `tree_path_dependent` expectations from the fitted trees. It does not silently pick evaluation records as background data. Tree SHAP support varies by estimator; arbitrary preprocessing pipelines are rejected because raw feature names may not match their transformed feature space. Pass the fitted final tree with its prepared transformed inputs, or use permutation importance on the complete pipeline.

Only a scalar regression output with an identity output scale is supported. A Poisson/log-link model, classifier, or model whose SHAP tree limit differs from its public prediction can fail the reconstruction check. This is intentional: an explanation of one output scale must not be presented as an explanation of another. Use permutation importance when that applies.

The result includes:

| Table | Interpretation |
|---|---|
| `global_importance` | Mean absolute and mean signed contributions, scaled by `error_scale`, across all explained rows. |
| `local_contributions` | Saved case positions, actual feature values, native and scaled per-feature contributions. |
| `predictions` | Base value, SHAP sum, raw prediction, fixed offset, reconstructed prediction, actual and residual. |

Every explanation must satisfy `base_value + shap_sum ≈ model.predict(features)` before it is returned. The check uses `rtol=1e-5, atol=1e-6` in native model target units. The original prediction, reconstructed prediction and additivity error remain available for audit.

### Inspect a specific or worst record

Random sampling is appropriate for a bounded population summary. The worst case **inside that sample** is not necessarily the worst case in the full population. Select explicit positions when investigating a known error:

```python
# SETUP LOGIC: Positional selection remains safe even with duplicate pandas index labels.
import numpy as np

# CORE LOGIC: STEP 1 — Locate the candidate's largest errors in the current paired population.
# Input: paired absolute errors=[1,5,2], requested case count=2.
# Output: worst_positions=[1,2], referring to the second and third paired records.
# Explanation: Rank errors, then retain their integer positions instead of joining on index labels.
# Trick: Stable sorting preserves source order for ties; these are review examples, not a representative sample.
worst_positions = np.argsort(
    -review.rows["__ae_candidate"].to_numpy(), kind="stable"
)[:2].tolist()

# EXPLANATION LOGIC: Explain exactly those current-candidate cases, not a new random sample.
worst_explanation = review.shap_values(
    population="paired", row_positions=worst_positions,
    max_rows=2, max_local_rows=2,
)

# PLOTTING LOGIC: Choose one saved paired-cohort position explicitly.
display(plot_shap_local(worst_explanation, row_position=worst_positions[0]))
```

With `population="paired"`, positions refer to the current comparison's paired-row order. With `population="model"`, positions refer to `review.data`, including its current filters, before invalid actual/offset rows are excluded. They are **not** pandas index labels or positions in an earlier unfiltered dataframe. An explicitly requested invalid row raises an error.

If you explain a model other than the current candidate, rank that model's errors or switch the selected pair first; the example deliberately uses the current candidate.

### Residual targets and fixed offsets

For a model trained on `actual_spread - anchor_spread`, `Model(..., offset="anchor_spread")` reconstructs the prediction:

```text
Tree expected value          2
Feature contributions    +3 -1
Raw residual prediction      4
Original row's anchor      100
Final spread prediction    104
Actual spread              103
Final error                  1
```

SHAP attributes the learned residual of 4; the anchor of 100 is shown separately. Permutation also scores the final prediction of 104 against the actual 103. If `anchor_spread` is both a feature and the reconstruction offset, shuffling its **feature input** does not change the original row's fixed reconstruction offset. That measures the learned model's reliance on the anchor input, not the effect of replacing the entire anchor pipeline.

`error_scale` converts losses and plotted contributions to the displayed error unit. The `predictions` reconstruction columns and local `shap_value` retain native target units; `shap_value_scaled`, global importance and residual errors use the scale. Merely writing `unit="bps"` changes a label, not numeric units.

## Slice-specific and two-model explanations

Filter the comparison first, then call its explanation methods. This reuses existing predictions and does not retrain. A slice explanation describes that slice only; it cannot establish that the feature is valuable everywhere.

For two-model comparisons, use `population="paired"` for both models, with the same `max_rows` and `random_state`. Check returned `row_position` values before comparing results. Features can differ between models, and SHAP values are relative to **each model's own expected value**, so subtracting two global SHAP bars is not an estimate of accuracy improvement. Compare permutation loss changes alongside each model's baseline error and the usual paired error tables.

The comparison proxies also check fresh raw predictions against the cached comparison predictions (`rtol=1e-7, atol=1e-8`). If you mutate/refit a retained estimator or alter its input processing, rebuild the comparison before explaining it. This prevents explaining a different model state from the one whose errors appear in the review.

The latest explanation for each method/model is attached to that comparison and included in its later report export. A new filter or selected pair creates a new comparison; compute explanations for that applied population so exported evidence stays consistent.

## What to investigate next

The implemented tools address model reliance, local prediction drivers and selected-population error diagnosis. Useful future extensions include **ALE/ICE curves** for nonlinear feature-response behavior, explicit **feature-distribution drift** between training and evaluation populations, and retrained **ablation studies** for incremental feature-set value. These are **not implemented** by the explanation API in this release. In particular, permutation is not a substitute for a fresh ablation model trained without the feature group.

## Method references

- [Scikit-learn: permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html): held-out evaluation, repeated shuffling and correlated-feature limitations.
- [Scikit-learn: correlated-feature example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html): why individually small importance can coexist with a useful feature group.
- [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html): output scales and feature-dependence assumptions.
- [LightGBM Booster prediction](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html): native contributions, expected-value column and iteration defaults.
