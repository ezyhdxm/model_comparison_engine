# Model Comparison Engine

Compare **any two scalar regression models** on the same evaluation records. Start with prediction columns, or pass two fitted estimators and their respective feature lists. Inspect overall errors, arbitrary slices, two-column heatmaps, paired statistical evidence, candidate residuals, coverage and date sensitivity. Export calculated HTML, CSV and complete PNG reports.

The package is independent of a particular dataset, industry, estimator library or training pipeline. It has no hardcoded business fields or thresholds. All shipped examples are generated synthetic data; no private records, models or historical result artifacts are included.

## Install and try

```bash
python -m pip install -e ".[notebook]"
python -m model_comparison_engine demo --output reports/demo
```

The command prints the generated HTML location. Demo model names explicitly identify their synthetic origin. These simulated gains do not establish real-world performance.

Open [model_comparison.ipynb](model_comparison.ipynb) for the interactive workflow, or use the API directly:

```python
# CONFIGURATION LOGIC: These are user-defined column names; the engine imposes no data-specific schema.
from model_comparison_engine import compare_predictions, InferenceConfig, Slice, show_comparison

# REPORTING LOGIC: Both predictions are scored on the same finite target/prediction records.
comparison = compare_predictions(
    data,
    actual="target", reference="prediction_a", candidate="prediction_b",
    reference_name="Model A", candidate_name="Model B",
    id_column="row_id", time_column="timestamp", entity_column="entity_id",
    unit="units", timezone="UTC",
)
# CONFIGURATION LOGIC: Auto uses date means when timestamps are configured, otherwise individual records.
inference = InferenceConfig(loss="absolute", unit="auto", correction="by", alpha=.05, min_units=10)
# REPORTING LOGIC: Reuse existing errors; each returned table is its own multiple-testing family.
paired_evidence = comparison.paired_test(inference=inference, min_count=30)
slice_evidence = comparison.cross_slice_test("segment", "category", inference=inference, min_count=30)
candidate_diagnostics = comparison.candidate_diagnostics(slices=["segment"], population="candidate")
# UI LOGIC: Choose columns, filters, bins, test settings and candidate population, then Apply.
panel = show_comparison(comparison, inference=inference, candidate_population="candidate")
# FILE IO LOGIC: Exact tables and complete figures are saved without printing a large table.
report_folder = comparison.export(
    "reports",
    slices=["segment", Slice("measure_1", [0,10,20,float("inf")], right=False)],
    interactions=[("segment", "category")],
    inference=inference, include_candidate=True, candidate_population="candidate",
)
```

Only target and two prediction columns are required. Record ID, timestamp and entity ID are optional metadata. An entity may have repeated records; it is distinct from a unique record ID. Default error units are `units`, and timezone is UTC. Add any column to your input for a new slice.

Paired tests use **candidate loss minus reference loss**, so a negative effect and t-statistic favor the candidate. Choose absolute or squared error, and record, date or entity units. Date/entity units first average losses within each unit and then receive equal weight. The default Benjamini–Yekutieli (BY) correction adjusts supported tests within each table; BH and unadjusted alternatives are explicit options. Stars indicate adjusted significance, and † marks insufficient support. Constant or near-constant differences receive no test or stars. [USAGE.md](docs/USAGE.md#paired-statistical-evidence) explains counts, assumptions and confidence intervals.

Candidate diagnostics default to all finite target/candidate-error records, including records with a missing reference. Their population can differ from the paired comparison: inspect the displayed counts or choose `population="paired"` for the same records. Residual plots, calibration, quantiles, worst slices/cases and dates describe candidate behavior without retraining or removing outliers. Non-significance does not establish equivalence; date grouping and out-of-fold predictions do not resolve serial correlation or repeated model selection.

## Two fitted estimators

```python
# CONFIGURATION LOGIC: Each estimator receives its own ordered feature list and optional preprocessing pipeline.
from model_comparison_engine import Model, compare_models

# INFERENCE LOGIC: predict is called once per estimator; fit is never called.
comparison = compare_models(
    data, actual="target",
    reference=Model("Model A", fitted_model_a, ["feature_1", "feature_2"]),
    candidate=Model("Model B", fitted_model_b, ["feature_1", "feature_2", "feature_3"]),
    id_column="row_id",
)
```

## Capabilities

- Same-record MAE, RMSE, P95/median absolute error, bias, tolerance rate, win/tie rates and relative improvement.
- Explicit prediction coverage, finite-error checks, missing metadata groups and low-support flags.
- Arbitrary categorical slices and fixed numerical bins; missing and out-of-range values remain separate.
- Two-column metric and sample-count heatmaps; population filters compose by intersection.
- Paired mean loss tests with record/date/entity units, confidence intervals, BY/BH correction and explicit unsupported states.
- Candidate-only error distribution, actual-versus-predicted calibration, residual quantiles and worst slices/cases, with an explicit population.
- Any saved model pair as reference/candidate; model names carry no special benchmark status.
- Identity-based prediction joins and explicit stage selection for long prediction tables.
- Optional per-model additive offsets when predictions are relative to a known reference value.
- Immutable local review bundles with configuration, filter history and input fingerprint.
- Notebook controls export the last **applied** result, not pending edits.
- Optional expanding/rolling walk-forward fitting with fold-local preprocessing, label-availability purging and an excluded final holdout.

## Optional walk-forward cross-validation

Use `walk_forward_compare` when you explicitly want to fit fresh models on earlier dates and score later dates. `compare_models`, `compare_predictions` and the review notebook remain prediction-only APIs.

```python
# SETUP LOGIC: A factory returns a fresh unfitted estimator or complete preprocessing pipeline per fold.
from sklearn.base import clone
from model_comparison_engine import TrainableModel, WalkForwardConfig, walk_forward_compare
# CONFIGURATION LOGIC: Choose chronological windows before reviewing gains; reserve the final dates separately.
folds = WalkForwardConfig(min_train_dates=30, validation_dates=5, embargo_dates=1,
                          n_splits=4, holdout_dates=10, holdout_embargo_dates=1)
# MODELING LOGIC: Only this explicit API fits; both models use the same available training labels and held-out records.
result = walk_forward_compare(
    data, target="target", time_column="timestamp", settings=folds,
    reference=TrainableModel("Model A", lambda: clone(model_a), ["feature_1", "feature_2"]),
    candidate=TrainableModel("Model B", lambda: clone(model_b), ["feature_1", "feature_2", "feature_3"]),
    id_column="row_id", entity_column="entity_id",
)
# FILE IO LOGIC: Save fold boundaries, fold losses, held-out predictions and a normal paired review.
folder = result.export("reports/cross_validation", slices=["segment"])
```

For these scikit-learn factories, install `python -m pip install -e ".[training]"`. Other estimators require only their own library and the `fit`/`predict` protocol. The [opt-in synthetic example](examples/walk_forward.py) shows explicitly configured categorical preprocessing. No category names, encoders or estimator parameters are inferred. If labels arrive later than their record timestamps, map `label_available_column`; unknown or boundary-late labels are purged from each training fold.

Read [USAGE.md](docs/USAGE.md) for data contracts, walk-forward policy, custom comparisons and interpretation. This release supports scalar regression, not classification metrics. Cross-validation does not evaluate the reserved final test or perform automatic model selection.
