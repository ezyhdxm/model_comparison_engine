# Model Comparison Engine

Compare **any two scalar regression models** on the same evaluation records. Start with prediction columns, or pass two fitted estimators and their respective feature lists. Inspect overall errors, arbitrary one-column slices, two-column heatmaps, missingness, coverage and date sensitivity. Export calculated HTML, CSV and complete PNG reports.

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
from model_comparison_engine import compare_predictions, Slice, show_comparison

# REPORTING LOGIC: Both predictions are scored on the same finite target/prediction records.
comparison = compare_predictions(
    data,
    actual="target", reference="prediction_a", candidate="prediction_b",
    reference_name="Model A", candidate_name="Model B",
    id_column="row_id", time_column="timestamp", entity_column="entity_id",
    unit="units", timezone="UTC",
)
# UI LOGIC: Choose columns, filters, numerical bins and interactions, then Apply.
panel = show_comparison(comparison)
# FILE IO LOGIC: Exact tables and complete figures are saved without printing a large table.
report_folder = comparison.export(
    "reports",
    slices=["segment", Slice("measure_1", [0,10,20,float("inf")], right=False)],
    interactions=[("segment", "category")],
)
```

Only target and two prediction columns are required. Record ID, timestamp and entity ID are optional metadata. An entity may have repeated records; it is distinct from a unique record ID. Default error units are `units`, and timezone is UTC. Add any column to your input for a new slice.

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
- Any saved model pair as reference/candidate; model names carry no special benchmark status.
- Identity-based prediction joins and explicit stage selection for long prediction tables.
- Optional per-model additive offsets when predictions are relative to a known reference value.
- Immutable local review bundles with configuration, filter history and input fingerprint.
- Notebook controls export the last **applied** result, not pending edits.

Read [USAGE.md](docs/USAGE.md) for data contracts, file-based runs, custom comparisons and interpretation. This release evaluates scalar regression; it does not implement classification metrics, train models or choose a validation/test partition.
