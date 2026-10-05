# Usage

## Inputs and evaluation scope

`compare_predictions` accepts a pandas DataFrame or a local CSV/Parquet path. Declare the actual target column and two different prediction columns. All three must represent compatible numerical values. No model name is privileged: reference and candidate are roles you choose for a particular comparison.

Optional fields are `id_column` (unique record identity), `time_column` and `entity_column` (repeated entity identity). Their names are arbitrary. Duplicate columns and names beginning with `__` are rejected because that prefix belongs to derived internal values. A supplied record ID must be unique and nonmissing. Rows with missing entity or time remain eligible for error metrics.

Use a typed DataFrame or Parquet for a complete source schema. CSV adapters preserve explicitly configured record/entity IDs and prediction join keys as strings, so `001` cannot silently match `1`. Other CSV fields follow pandas inference in the Python API. `read_data(path, string_columns=["custom_id"])` preserves additional text columns. Notebook file loading preserves all CSV fields as strings; numerical targets, predictions and binned measurements are converted explicitly during comparison. DataFrame dtypes remain caller-owned. When mixing CSV paths and DataFrames for a join, their key types must agree.

Prepare your evaluation partition upstream. Do not mix training and held-out predictions in one comparison. This package reports the data you supply; it does not prove that targets, features or offsets were available at prediction time.

## Paired metrics and units

Both models are evaluated on records where actual and both reconstructed predictions are finite and both scaled errors are finite. The report separately counts supplied records, valid targets, availability for each model, paired records and exclusions. Individual missingness categories can overlap; only total minus paired is an exclusive exclusion count. No missing prediction is filled with zero.

Signed error is `(prediction - actual) * error_scale`. Positive bias means overprediction. All loss metrics use the declared display unit. `error_scale` must be positive and finite; changing the unit label alone changes no numbers. `tolerance` is expressed in the displayed error unit. The package does not convert between incompatible target definitions.

MAE and RMSE are record weighted. P95 is calculated from individual absolute errors, not averaged subgroup quantiles. Candidate-minus-reference MAE/P95 below zero indicates improvement; positive relative MAE improvement indicates improvement. Relative improvement is undefined when reference MAE is zero. Small cells are flagged rather than removed. Overlapping slices must not be added together.

When an estimator predicts a residual, supply `reference_offset="known_level"` and/or `candidate_offset="other_known_level"`. The corresponding column is added to that model's prediction before evaluation. Each model can use a different offset; the actual target must already be on the final scale. `Model(..., offset="known_level")` supplies the same behavior with fitted estimators. Offsets remain attached to model identities when the notebook reference/candidate selectors are swapped.

## Fitted estimators

`Model(name, estimator, features, offset=None)` wraps any fitted object with `predict`. Feature order is explicit and may differ between the two models. Predictions must contain one scalar per input record; an `(n,1)` array is accepted, multi-output arrays are rejected. A pandas prediction must preserve the exact input index. Array predictions rely on the estimator's record-order contract.

The engine performs no fitting, imputation, encoding or model deserialization. Include any required preprocessing inside the fitted estimator/pipeline you pass. Saved predictions avoid inference entirely.

## Join predictions by identity

```python
# DATA ADAPTER LOGIC: Join a separate prediction table by unique identity, retaining unmatched evaluation records.
from model_comparison_engine import attach_predictions, from_long_predictions
data = attach_predictions(data, prediction_table, on="row_id", columns={"output_a":"prediction_a", "output_b":"prediction_b"})
# DATA ADAPTER LOGIC: A long prediction table has one row per model/record/stage.
wide = from_long_predictions(data_without_predictions, long_predictions,
                             on="row_id", model="model", value="prediction",
                             stage_column="stage", stage="Validation")
```

The join rejects duplicate/missing keys and accidental column overwrites. Long tables with multiple stages require an explicit stage. Duplicate model/record rows are errors, never silently averaged. A missing joined prediction remains visible in coverage. Key types must already agree.

## Custom slices, filters and heatmaps

```python
# CONFIGURATION LOGIC: Boundaries are user choices, not fitted to the most favorable error result.
from model_comparison_engine import Slice
measure = Slice("measure_1", [0,10,20,float("inf")], labels=["low","middle","high"], right=False)
# REPORTING LOGIC: Return exact computed tables for programmatic use; printing is optional.
one_dimension = comparison.slice(measure, min_count=50)
two_dimensions = comparison.cross_slice("segment", measure, min_count=50)
# REPORTING LOGIC: Chained filters create an intersection on the original supplied population.
subset = comparison.filter("measure_1", minimum=10).filter("category", values=["A","B"])
# FILE IO LOGIC: Export chosen slices and interactions with full filter provenance.
folder = subset.export("reports", slices=[measure], interactions=[("segment",measure)], min_count=50)
```

`right=False` uses `[a,b)` numerical bins; `right=True` uses `(a,b]` and includes the lowest endpoint in the first bin. Missing/nonfinite values and finite out-of-range values form distinct groups. Categorical top groups are selected by support frequency, with stable lexical tie handling; remaining categories become a separate Other group without colliding with literal labels.

The package supplies only generic time defaults. Define your category columns and numerical boundaries explicitly with `Slice` objects for repeatable defaults. Any retained metadata/feature column can be selected. Heatmaps show both metric and support; an undefined metric can coexist with a nonzero sample count.

## Notebook interface

Install the `notebook` extra, open `model_comparison.ipynb`, and run its cells. It uses synthetic records by default. To inspect your results, replace the synthetic table with your DataFrame or path and declare prediction columns.

```python
# UI LOGIC: More than two saved models can be offered; select any two without rerunning inference.
from model_comparison_engine import show_comparison
panel = show_comparison(data, actual="target",
    predictions={"Model A":"prediction_a", "Model B":"prediction_b", "Model C":"prediction_c"},
    id_column="row_id", time_column="timestamp", entity_column="entity_id",
    default_slices=[Slice("segment"), measure])
```

Select the reference, candidate, slice column, optional second column, bins, metric and minimum support. Two optional filter rows form an intersection. A filter with no condition is a no-op. Click **Apply comparison** to update results. Editing controls alone changes neither the applied result nor an export. Exports use the recorded applied model pair, filters, bins, metric and support threshold.

The UI does not fit models. Swapping among columns reuses saved predictions. A failed Apply retains the previous successful result and identifies the error. No timestamp is required; date analysis is then unavailable.

## Time and stability

The default timezone is UTC. Naive timestamps are interpreted in the configured timezone; aware timestamps are converted to it. Ambiguous/nonexistent naive DST times are unassigned and counted as missing temporal metadata. Mixed naive/aware inputs are rejected. Missing dates do not remove otherwise valid predictions.

Daily metrics use actual date order. Leave-one-date-out sensitivity removes saved error records for each date, without retraining. Its range is descriptive; it is not a confidence interval, independence guarantee or a significance test. A few dates or sparse groups can produce unstable conclusions.

## File-based report

Copy [examples/config.json](../examples/config.json), edit the input path and column mappings, and run:

```bash
python -m model_comparison_engine compare --config examples/config.json
```

Relative input/output paths resolve beside the config. Optional `--output` is relative to the current working directory. Remove optional metadata mappings if those columns do not exist. Open the printed `report.html` path. Each export creates a new directory containing exact aggregate CSVs, complete PNG figures and `review.json` with settings, filters, coverage and a full-precision input fingerprint.

## Repository contents and output labels

All bundled records are generated synthetic examples with neutral labels. The repository contains code, documentation, configuration examples and an output-free notebook; it includes no private input files, fitted models, predictions, caches or research-result artifacts. Generated outputs are ignored by Git.

Reports on your own data retain the labels and groups you supplied. The package does not automatically anonymize user datasets or provide privacy guarantees. Aggregates can still expose sensitive group names or small cells; rename/remove those fields upstream when an anonymous exported report is required. The synthetic demo's results illustrate behavior only.
