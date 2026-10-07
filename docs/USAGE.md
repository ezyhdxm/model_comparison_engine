# Usage

## Updating an already-running notebook

Replace or update the complete package folder before reloading; copying only `ui.py` can mix incompatible versions. Python also retains imported classes and existing widget callbacks in memory. An error such as `TradeViewConfig.__init__() got an unexpected keyword argument 'point_view'` indicates that the newer controls reached an older constructor.

After updating the files, run this cell. It reloads only diagnostic code and preserves your data, predictions and fitted models:

```python
# SETUP LOGIC: Reload dependencies before their UI/report consumers; preserve notebook data and model variables.
import importlib
for module in ("trade_view", "trade_plots", "report", "ui"):
    importlib.reload(importlib.import_module(f"model_comparison_engine.{module}"))
import model_comparison_engine as mce
importlib.reload(mce)
from model_comparison_engine import TradeViewConfig, show_comparison
```

Then rerun the cell that creates the comparison panel and click **Apply comparison on the newly created panel**. Existing widgets keep their old callbacks, so clicking an older panel may repeat the error. Do not rerun training or prediction generation, and do not restart the kernel.

Version 0.5.2 rebuilds retained trade configurations with the currently loaded class, preserving frequency, sampling seed, metadata mappings and advanced limits. Older objects receive defaults for newly added options. If the error persists after reloading, inspect `mce.__version__` and `mce.__file__` to confirm that Python is loading the folder you updated.

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

For example, actual `105`, residual prediction `2` and offset `100` reconstruct a prediction of `102`; at `error_scale=1` the signed error is `-3` and the absolute loss is `3`. An offset is added before error scaling, and must be known at prediction time. Candidate diagnostics use the same reconstruction, even when that row has no reference prediction.

## Paired statistical evidence

`InferenceConfig(loss="absolute", unit="auto", correction="by", alpha=.05, min_units=10)` configures a two-sided paired test of zero mean **candidate loss minus reference loss**. Absolute loss uses the displayed error unit; squared loss uses its square and tests a mean squared-error difference, not an RMSE difference. A negative effect and t-statistic favor the candidate. The method tests paired loss differences, not signed prediction errors or independent model samples. The t statistic and pointwise confidence interval follow the related-sample mean-difference convention documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html).

For 100 paired records, the input to a record-level test contains 100 candidate-minus-reference loss differences. If every reference absolute error is `2`, 50 candidate errors are `1` and 50 are `2`, those differences are 50 values of `-1` and 50 of `0`, with mean `-0.5`. Treating the two error arrays as 200 independent observations loses their pairing. Repeated dates or entities can make even those 100 paired differences dependent.

| Test unit | Values supplied to the test | Weighting and exclusions |
|---|---|---|
| `record` | One loss difference per finite paired record | Every tested record has equal weight |
| `date` | One mean loss difference per configured local date | Every observed date has equal weight; rows with missing dates are excluded from the test |
| `entity` | One mean loss difference per configured entity | Every observed entity has equal weight; rows with missing entity IDs are excluded from the test |
| `auto` | Date means when a timestamp column is configured; otherwise records | It does not silently fall back to records when configured dates are missing |

An explicit `date` or `entity` unit requires the corresponding metadata mapping. Equal unit weighting can change the estimated effect: 90 records on one date with difference `-1` and 10 on another with difference `+1` give a record mean of `-0.8`, but the two date means average to `0`. There are 100 records and only 2 date units, so the default `min_units=10` does not permit a test. Ordinary MAE/RMSE tables remain record weighted.

Each returned table contains these fields in addition to its group labels and recorded inference settings:

| Fields | Interpretation |
|---|---|
| `row_count`, `n`, `paired_n` | Original paired rows in the cell; `n` and `paired_n` count finite paired losses for the selected loss |
| `tested_n`, `excluded_unit_rows` | Rows with finite paired losses and known test-unit labels; finite-loss rows excluded for missing unit metadata |
| `unit_count` | Number of record/date/entity values supplied to the test |
| `mean_loss_difference` | Mean candidate-minus-reference loss across the selected units |
| `t_statistic`, `p_value`, `q_value` | Signed t statistic, raw two-sided p value, and value adjusted within this table |
| `ci_low`, `ci_high` | Pointwise `(1-alpha)` confidence interval for the selected-unit mean; no multiplicity adjustment |
| `significance`, `significant` | Stars at adjusted q ≤ .05/.01/.001; a separate boolean decision at the configured alpha |
| `low_support`, `status` | Support warning and explicit reason when testing is unavailable |

Both the minimum usable record count (`tested_n >= min_count`, default 30) and minimum unit count must pass. `tested_n` counts eligible records even when support or variance rules subsequently prevent testing. A cell with any nonfinite loss is untested rather than silently dropping the overflowed record. Unsupported cells, too few usable units, nonfinite calculations, and zero or near-zero variance produce no valid p/q value or significance stars. A deterministic difference is not displayed as infinite statistical certainty. Read the `status` and counts rather than interpreting a blank test as a zero effect. Stars in significance figures denote adjusted significance, while † denotes low support in metric figures; support markers are not hypothesis-test results.

The default `correction="by"` uses Benjamini–Yekutieli false-discovery-rate adjustment across finite supported tests in **one returned table**. It is a conservative default for dependence between overlapping slice tests. `"bh"` selects Benjamini–Hochberg, appropriate under independence or suitable positive dependence; `"none"` leaves `q_value` equal to the raw p value. These dependence distinctions are described in the [statsmodels FDR documentation](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.fdrcorrection.html). An overall table, each one-way slice table and each interaction table are separate families. Correction does not extend across multiple tables, model pairs, filters, repeated Apply actions or exploratory searches.

```python
# SETUP LOGIC: Public configuration and result APIs require no training library.
from model_comparison_engine import InferenceConfig, Slice, compare_predictions
from model_comparison_engine.inference_plots import significance_heatmap
# CONFIGURATION LOGIC: Declare the scored columns, units and grouping before inspecting results.
comparison = compare_predictions(data, actual="target", reference="prediction_a", candidate="prediction_b",
    reference_name="Model A", candidate_name="Model B", id_column="row_id",
    time_column="timestamp", entity_column="entity_id", error_scale=1, unit="units", timezone="UTC")
inference = InferenceConfig(loss="absolute", unit="date", correction="by", alpha=.05, min_units=10)
measure = Slice("measure_1", [0,10,20,float("inf")], labels=["low","middle","high"], right=False)
# REPORTING LOGIC: These methods pair existing losses and preserve the chosen slice boundaries.
overall_test = comparison.paired_test(inference=inference, min_count=30)
segment_tests = comparison.slice_test("segment", min_count=30, top_n=20, inference=inference)
binned_tests = comparison.slice_test("measure_1", bins=[0,10,20,float("inf")],
    labels=["low","middle","high"], right=False, min_count=30, inference=inference)
interaction_tests = comparison.cross_slice_test("segment", measure, min_count=30, top_n=12, inference=inference)
# PLOTTING LOGIC: Effect significance and support use separate annotations in the complete figure.
figure = significance_heatmap(interaction_tests, title="Model B vs Model A", unit_label=comparison.unit)
# FILE IO LOGIC: Export this exact test configuration alongside every chosen metric table.
folder = comparison.export("reports/paired_review", slices=["segment",measure],
    interactions=[("segment",measure)], min_count=30, metric="mae_delta", inference=inference,
    include_candidate=True, candidate_population="candidate", candidate_top_n=20)
```

`slice_test(column, bins=None, labels=None, min_count=30, top_n=20, right=True, inference=None)` accepts a column or `Slice`. `cross_slice_test(x, y, x_bins=None, y_bins=None, min_count=30, top_n=12, x_right=True, y_right=True, inference=None)` accepts a column or `Slice` on each axis. `paired_test(inference=None, min_count=30)` provides the overall row. Omitting inference uses the defaults above; `comparison.inference_config(inference)` returns the resolved configuration, including the unit chosen for `auto`.

The t approximation requires adequate independent units and finite, suitably behaved loss differences; grouping by date or entity does not guarantee this. Dates can remain serially correlated, entities can share shocks, and out-of-fold predictions can share training data. This release does not implement HAC standard errors, block bootstrap, or multiway dependence correction. BY addresses dependence across tests, not invalid p values caused by dependence within a test. Time-aware cross-validation prevents specified training leakage but does not solve these inference assumptions. Non-significance, especially with low support, is not proof of equivalence or no practical difference. Inspect effect sizes, intervals, coverage and loss tails alongside significance; confirm exploratory findings on a predeclared evaluation set.

## Candidate diagnostics and population

`comparison.candidate_rows(population="candidate")` includes all records with a finite actual, reconstructed candidate prediction and scaled candidate error, after the comparison's population filters. Missing reference predictions do not remove these records. `population="paired"` uses exactly the paired comparison rows. Thus a supplied population of 100 valid targets/candidate predictions with 10 missing reference predictions has 100 candidate records and 90 paired records; a candidate-only MAE is not directly comparable to the reference MAE on those 90 rows.

```python
# SETUP LOGIC: Candidate plots use the same stored predictions and error scaling as the comparison.
from model_comparison_engine.diagnostic_plots import candidate_figure
# REPORTING LOGIC: Choose the population explicitly; no prediction or reference value is imputed.
candidate_rows = comparison.candidate_rows(population="candidate")
diagnostics = comparison.candidate_diagnostics(slices=["segment",measure],
    population="candidate", min_count=30, top_n=20, bins=10)
paired_diagnostics = comparison.candidate_diagnostics(slices=["segment"], population="paired")
# PLOTTING LOGIC: Residual distributions and actual-versus-predicted patterns describe this population.
candidate_plot = candidate_figure(candidate_rows, diagnostics, name=comparison.candidate_name,
    unit=comparison.unit, error_scale=comparison.config["error_scale"])
```

The returned dictionary contains `summary`, `calibration`, `residual_quantiles`, `worst_slices`, `worst_cases` and `daily` DataFrames. `slices=None` uses the comparison's generic default slices. `bins` sets the requested number of prediction-quantile calibration bins; tied boundaries can reduce the actual number. Calibration means use native target units, while residuals use the configured error scale. `min_count` flags low support and `top_n` bounds the worst-slice and worst-case tables. Slice category limits come from each `Slice.top_n`. Worst slices rank supported groups first by MAE and then include low-support groups if space remains. Error shares use the full population absolute-error total; tail contributions use errors strictly above the population P95. Shares from overlapping slice specifications must not be added together. Summary and report captions disclose the population and coverage. Signed residuals use prediction minus actual, so positive residuals indicate overprediction; quantiles and worst cases retain the tails rather than removing outliers. Worst-case rows include the supplied record ID when mapped, plus `__source_position` (position within the filtered source population) and `__source_index` (the original index label). Diagnostic rankings help inspect failure modes, but carry no automatic hypothesis tests or model-selection decision. Scatter displays use a bounded deterministic sample; aggregate tables use the full selected population.

`export(..., inference=None, include_candidate=True, candidate_population="candidate", candidate_top_n=20)` includes the paired inference and candidate report by default. Set `include_candidate=False` to omit candidate outputs. CSVs provide full precision; HTML and PNGs display rounded values. The notebook Candidate tab exposes the same population choice and tables. These APIs never train, recalibrate, impute, filter by residual size, or mutate your predictions.

## Optional residual timing

Start with daily bias/MAE and bin support. Then review fixed-clock lag correlations and within-entity event lags; they measure different notions of adjacency. `TemporalConfig(frequency="1D")` uses fixed 24-hour UTC bins, while lag one in the event table is the next distinct timestamp for the same entity and can span a long inactive period. Inspect median/P90 elapsed gaps and consider one entity before attributing pooled correlation to persistence.

```python
# SETUP LOGIC: Optional timing analysis reuses the comparison's existing residuals.
from model_comparison_engine import TemporalConfig
# CONFIGURATION LOGIC: Begin with daily bins and declare the diagnostic population explicitly.
timing = TemporalConfig(frequency="1D", signal="bias", rolling_bins=10, max_lag=20)
# REPORTING LOGIC: Support, gaps and unavailable states accompany the exploratory calculations.
timing_tables = comparison.temporal_diagnostics(timing, population="candidate")
# FILE IO LOGIC: The report records this exact configuration and exports full timing tables and figures.
folder = comparison.export("reports/timing_review", slices=["segment"],
    temporal=timing, temporal_population="candidate")
```

The returned tables are `summary`, `series`, `autocorrelation`, `spectrum`, `change_points` and `event_autocorrelation`. Empty bins remain missing rather than zero. Spectral peaks and mean-shift candidates are descriptive, exploratory results; they do not carry calibrated significance, fix dependence in paired tests, select a model or trigger refitting. `temporal=None` in `export` omits this section. See [TIME_SERIES.md](TIME_SERIES.md) for bin weighting, support thresholds, period bounds and change-candidate interpretation.

## Fitted estimators

`Model(name, estimator, features, offset=None)` wraps any fitted object with `predict`. Feature order is explicit and may differ between the two models. Predictions must contain one scalar per input record; an `(n,1)` array is accepted, multi-output arrays are rejected. A pandas prediction must preserve the exact input index. Array predictions rely on the estimator's record-order contract.

`compare_models` performs no fitting, imputation, encoding or model deserialization. Include any required preprocessing inside the fitted estimator/pipeline you pass. Saved predictions avoid inference entirely. The separate opt-in `walk_forward_compare` API below fits models when explicitly called.

Version 0.5 retains fitted models in memory for explicitly requested explanations. Export never serializes estimators.
Changing filters or selecting a cached model pair does not call `predict` again. Explanation buttons do call
`predict` on bounded samples and show progress. A changed estimator or reconstruction offset must be re-evaluated
before explaining its old cached predictions. See [model explanations](EXPLANATIONS.md).

## More than two models

Pass any number of saved prediction columns using readable display names:

```python
# SETUP LOGIC: A mapping separates readable model names from DataFrame column names.
from model_comparison_engine import compare_prediction_set, show_comparison
# CONFIGURATION LOGIC: Offsets are optional and belong to model identities, not reference/candidate slots.
review = compare_prediction_set(data, actual="observed",
    predictions={"Model A":"prediction_a", "Model B":"prediction_b", "Model C":"prediction_c"},
    reference="Model A", candidate="Model B", time_column="timestamp", entity_column="instrument_id",
    prediction_offsets={"Model C":"known_level"})
# UI LOGIC: Any pair can be selected in the dropdowns; click Apply to change the evaluated pair.
panel = show_comparison(review)
# REPORTING LOGIC: The programmatic equivalent also reuses saved predictions.
other_pair = review.select_models("Model B", "Model C")
```

The direct UI input `show_comparison(data, actual="observed", predictions={...}, prediction_offsets={...})`
also accepts many prediction columns. If model predictions are stored in a separate table, use
`attach_predictions(..., on="record_id")` first. Do not assume two separate frames share row order.
Coverage is **pair-specific**, not the intersection of every supplied model. Switching pairs can change
the paired cohort when prediction availability differs; review the coverage numbers before comparing gains.

For fitted models:

```python
# SETUP LOGIC: Each fitted model keeps its own ordered inputs and optional additive offset.
from model_comparison_engine import Model, compare_model_set
# INFERENCE LOGIC: Each fitted estimator predicts once; no model is trained.
review = compare_model_set(data, actual="observed", models=[
    Model("Model A", fitted_a, features_a),
    Model("Model B", fitted_b, features_b),
    Model("Model C", fitted_c, features_c, offset="known_level")],
    reference="Model A", candidate="Model C", time_column="timestamp", entity_column="instrument_id")
```

## Friendly filters and option meanings

The workbench has four optional filter rows. Choose a column, type a search term, select one or more
suggested categories, and add them to the filter. Suggestions are a convenience: their bounded list
does not restrict which values the full-data filter can match. Select **Contains**, **Starts with** or
**Ends with** for partial text, and set case sensitivity explicitly. Punctuation is literal, not regex.
Semicolon-separated values in one row are OR; different rows are AND. Numerical bounds and categorical
matching cannot be mixed in one row. Apply is required before results and exports change.

```python
# CONFIGURATION LOGIC: Literal case-insensitive substrings can identify a family of category names.
subset = review.filter("issuer", values=["alpha", "beta"], match="contains", case_sensitive=False)
subset = subset.filter("quantity", minimum=1_000_000)
```

Identifier-like names such as `instrument_id`, `CUSIP`, `ISIN` and `entity_id` appear early in the
Entity ID menu. Ranking only helps discovery; it does not redefine your chosen identity. Entity ID is
a repeatable instrument/entity grouping, while Record ID must uniquely identify a row.

| Control | Meaning |
|---|---|
| Reference | The comparator for this review; it need not be a special BASE model. |
| Candidate | The model being investigated. Either supplied model can take either role. |
| Candidate diagnostic sample | All valid candidate records, or just the paired cohort; this never changes the paired comparison. |
| Correction: BY | Benjamini–Yekutieli false-discovery-rate correction; allows arbitrary dependence among the slice tests when their p-values are valid. Conservative default. |
| Correction: BH | Benjamini–Hochberg correction; less conservative, assuming independence or appropriate positive dependence among tests. |
| Correction: None | Raw p-values; no multiple-testing control. |
| Alpha | Threshold applied to adjusted q-values; statistical significance alone does not measure business value. |

Each returned table is a separate correction family. None of these corrections fixes serially dependent
test units or repeated exploratory model selection. See [SciPy's method reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

## Intraday and prediction-point review

```python
# CONFIGURATION LOGIC: Metadata roles are explicit; the engine never guesses side or quote conventions.
from model_comparison_engine import TradeViewConfig
trade_view = TradeViewConfig(frequency="auto", min_count=10, max_points=2000,
    side_column="side", counterparty_column="counterparty_type", dealer_column="dealer_id",
    quantity_column="quantity", point_view="residual", focus_entity=None)
# UI LOGIC: Applied filters also restrict the point plots and intraday metrics.
panel = show_comparison(review, trades=trade_view)
# REPORTING LOGIC: Complete bin statistics and the exact displayed point sample remain available.
trade_tables = review.trade_diagnostics(trade_view)
# FILE IO LOGIC: Include full PNGs, bin CSVs and offline interactive plots when Plotly is installed.
report_folder = review.export("reports", trades=trade_view)
```

Auto resolution uses the calendar span: up to three days uses 30-minute bins, up to fourteen uses
hourly bins, and longer spans use daily bins. Explicit intervals are available. Empty/unsupported bins
remain gaps, including nights and weekends; they are not drops to zero error. Record-count bars show
support for each interval. Statistical test units are unchanged: finer plotting intervals do not manufacture
independent observations. Prediction-versus-actual plots also work without timestamps; time plots require them.
Point sampling affects only the plots; bin metrics use all paired records. Hover preserves exact
metadata and prediction values. The Trade points tab shows compact coverage cards instead of a wide
one-row metadata table; full summary fields remain available through the API and export.

**Point view defaults to residuals**: prediction minus actual in the configured error unit, with zero
meaning an exact prediction. This avoids a pooled level plot looking impressive simply because bonds
trade at very different spread levels. Choose `point_view="within_entity"` to subtract each entity's
same mean actual value from both actual and predictions, then apply `error_scale`. The mean uses the
full focused paired cohort before sampling. This removes average level, not volatility; it is a
retrospective diagnostic using realized outcomes, not a predictive feature. `point_view="level"`
retains original target units for level inspection.

The **Focus entity (trade tab only)** selector retains exact source ID values and includes all known
nonmissing IDs, with supplied counts before Apply and last-applied paired counts afterward. Set
`focus_entity` to the exact source value, such as `"Instrument A"` or numeric `101`, or leave it `None`
for the full applied paired cohort. This changes only the trade charts and their interval summaries;
other comparison tabs retain the globally filtered population. Exports preserve the focus and view
settings. Coverage names the actual target column and distinguishes sampled from drawable records.

In the point plots, **side controls color**. Map the prepared side column with `side_column`, for example
`side_column="side"`. Supplied `D`/`B`/`S` codes use purple/blue/orange consistently; the engine does not
translate those codes into a presumed customer/dealer convention. Actual/reference/candidate are
separated clearly: residual mode uses x markers and a zero-error line, with separate panels for each model.
Within-entity and level modes use hollow circles for actual values and x markers for predictions.
Quantity controls marker size. Dealer and counterparty values
remain exact hover metadata. When no side column is supplied, dealer and then counterparty provide
fallback category colors. See [trade view details](TRADES.md).

The Time series tab now reports **Observed / Meaning / Next check** alongside computed coverage,
autocorrelation, spectrum and mean-shift results. These explanations describe actual returned tables;
they do not claim forecasting gains or calibrated significance for a scanned spectral peak.

## Walk-forward cross-validation

`walk_forward_splits(data, time_column, settings, timezone='UTC')` is a model-independent splitter. `WalkForwardConfig` counts **observed local dates**, so one busy date cannot straddle Train and Validation. It rejects invalid feature timestamps and overlapping validation windows. Nothing reads targets or fits a model while planning folds.

| Setting | Meaning |
|---|---|
| `min_train_dates` | Minimum scheduled training dates before the first fold |
| `validation_dates` | Number of dates in each complete held-out block |
| `embargo_dates` | Excluded dates immediately before each validation block |
| `step_dates` | Distance between validation starts; default is validation length; may be larger but cannot overlap |
| `max_train_dates` | `None` for expanding history; a positive cap for a rolling training window |
| `n_splits` | Keep up to this many latest complete folds; actual count is returned explicitly |
| `holdout_dates` | Reserve this many final dates from **all** folds; default zero means no automatic final reservation |
| `holdout_embargo_dates` | Additional excluded buffer before a nonzero final holdout |
| `label_available_column` | Optional timestamp at which each target label became observable; otherwise label availability is the caller's assumption |

Partial final validation blocks remain unscored. With dates January 1–12, 2026, `min_train_dates=3`, `validation_dates=2`, `embargo_dates=1` and `holdout_dates=2`, the folds validate January 5–6, 7–8 and 9–10. January 11–12 never enter fitting or validation. `max_train_dates=3` makes the last fold train January 5–7 instead of January 1–7.

If a label availability mapping is supplied, training records whose label timestamp is missing or is at/after the first validation date's midnight are excluded. `train_rows` reports rows after this purge; `scheduled_train_rows` and `label_purged_rows` disclose the difference. Date boundaries describe the scheduled window before label purging. Both estimators subsequently use the same finite-target training records; their feature-specific missing values are not silently filtered.

```python
# SETUP LOGIC: clone copies estimator configuration without carrying fitted sklearn state into another fold.
from sklearn.base import clone
from model_comparison_engine import TrainableModel, WalkForwardConfig, walk_forward_compare
# CONFIGURATION LOGIC: The caller declares features, preprocessing and temporal policy before looking at results.
settings = WalkForwardConfig(min_train_dates=30, validation_dates=5, embargo_dates=1,
    max_train_dates=60, n_splits=4, holdout_dates=10, holdout_embargo_dates=1,
    label_available_column="label_known_time")
reference = TrainableModel("Model A", lambda: clone(model_a), ["feature_1", "category"])
candidate = TrainableModel("Model B", lambda: clone(model_b), ["feature_1", "feature_2", "category"])
# MODELING LOGIC: Each factory returns a fresh full preprocessing/model pipeline, fitted only on that fold's training rows.
result = walk_forward_compare(data, "target", reference, candidate, time_column="timestamp",
    settings=settings, id_column="row_id", entity_column="entity_id", timezone="UTC", unit="units")
# REPORTING LOGIC: The usual arbitrary slices and intersections apply to held-out predictions from every fold.
summary = result.comparison.summary()
by_group = result.comparison.slice("segment")
per_fold = result.fold_metrics
# FILE IO LOGIC: Export folds.csv, fold_metrics.csv, oof_predictions.parquet and walk_forward.json with the paired report.
folder = result.export("reports/cross_validation", slices=["segment"], interactions=[("segment", "category")])
```

Each `TrainableModel` accepts `name`, `factory`, ordered `features`, optional `transformer_factory` and optional additive `offset`. Factories must return **fresh unfitted** objects. Reusing the same object across folds/models is rejected. A transformer implements `fit(X, y)` and `transform(X)`; it is fitted only on the training rows and transforms validation without reading validation targets. A complete estimator pipeline can instead own its preprocessing. No category detection, imputation, encoding, scaling or model parameters are inserted. Models must implement `fit(X, y)` and return one prediction per row from `predict(X)`; pandas outputs must preserve row indexes, and array outputs rely on the estimator's order contract.

The training target is shared by the two models. `actual=` can specify a separate scoring level when the supplied target is a residual; each model's `offset` is added using the existing paired-comparison convention. Use factories that contain no external evaluation data, fitted state or early-stopping holdout. Engine temporal controls cannot prove that an arbitrary factory or precomputed feature is causal.

`WalkForwardResult` holds `comparison`, `predictions`, `folds`, `fold_metrics` and `configuration`. Each source row is scored at most once and keeps `cv_row_position` and `cv_fold`. Pooled losses weight records; per-fold losses disclose drift and differing support. Exports describe the factory names and features, not a serialized executable training recipe: retain your factory code, library versions and parameters with the experiment.

Cross-validation is development evidence, not a final untouched test or an independence guarantee for subsequent paired tests. Later folds may train on labels from earlier validation dates once those labels are available. Repeated model/slice selection on CV still requires separate final confirmation. For delayed or overlapping labels, supply label availability and a suitable embargo; those controls do not infer horizon length. Reserve final dates explicitly, or provide a development-only DataFrame. The API never evaluates or refits on the reserved final holdout and does not choose a winning model automatically. The comparison notebook stays prediction-only; [examples/walk_forward.py](../examples/walk_forward.py) contains a separate disabled-by-default training example.

Out-of-fold scoring is **retrospective**: `label_available_column` gates each fold's training, but a held-out outcome can be scored even if it became available after a later decision cutoff. Before using these scores to freeze a model for a final holdout, restrict selection to outcomes known before that decision time. The generic comparison API makes no automatic selection and does not infer that cutoff for you.

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
from model_comparison_engine import show_comparison, TemporalConfig
panel = show_comparison(data, actual="target",
    predictions={"Model A":"prediction_a", "Model B":"prediction_b", "Model C":"prediction_c"},
    id_column="row_id", time_column="timestamp", entity_column="entity_id",
    default_slices=[Slice("segment"), measure], inference=InferenceConfig(),
    candidate_population="candidate", temporal=TemporalConfig(frequency="1D"))
```

The workbench separates data mapping and view choices from expandable inference, temporal and filter controls. A pending-edit badge distinguishes current controls from the applied review. Select the reference, candidate, slice column, optional second column, bins, metric and minimum record support. Set the loss, inference unit, correction, alpha and minimum independent units, plus the candidate diagnostic population. The Slices tab contains metric and significance views; the Candidate tab contains residual plots and all six diagnostic tables, including worst-slice support and error contribution. Enable **Include time-series diagnostics** before Apply to add the **Time series** tab; start with Daily. Passing `temporal=TemporalConfig(...)` enables and initializes those controls. The candidate population and population filters also apply to the temporal view. Two optional filter rows form an intersection. A filter with no condition is a no-op. Click **Apply comparison** to update results. Editing controls alone changes neither the applied result nor an export. Exports use the recorded applied model pair, filters, bins, metric, support thresholds, resolved inference settings, candidate population, top-group limit and enabled temporal configuration. Each comparison table remains its own correction family.

The UI does not fit models. Swapping among columns reuses saved predictions. A failed Apply retains the previous successful result and identifies the error. Candidate diagnostics remain available when the chosen candidate has valid records but the reference has none. No timestamp is required; date analysis is then unavailable and automatic inference uses records.

## Time and stability

The default timezone is UTC. Naive timestamps are interpreted in the configured timezone; aware timestamps are converted to it. Ambiguous/nonexistent naive DST times are unassigned and counted as missing temporal metadata. Mixed naive/aware inputs are rejected. Missing dates do not remove otherwise valid predictions.

Daily metrics use actual date order. Leave-one-date-out sensitivity removes saved error records for each date, without retraining. Its range is descriptive; it is not a confidence interval, independence guarantee or a significance test. A few dates or sparse groups can produce unstable conclusions.

## File-based report

Copy [examples/config.json](../examples/config.json), edit the input path and column mappings, and run:

```bash
python -m model_comparison_engine compare --config examples/config.json
```

Relative input/output paths resolve beside the config. Optional `--output` is relative to the current working directory. Remove optional metadata mappings if those columns do not exist. Open the printed `report.html` path. Each export creates a new directory containing exact aggregate CSVs, complete PNG figures and `review.json` with settings, filters, coverage and a full-precision input fingerprint. The example config enables daily temporal diagnostics; remove its `temporal` object to omit them. Charts wrap full category names and choose date ticks from the rendered label widths. Large heatmaps stack their effect/support panels, and dense category figures remain tall so labels stay readable.

## Repository contents and output labels

All bundled records are generated synthetic examples with neutral labels. The repository contains code, documentation, configuration examples and an output-free notebook; it includes no private input files, fitted models, predictions, caches or research-result artifacts. Generated outputs are ignored by Git.

Reports on your own data retain the labels and groups you supplied. The package does not automatically anonymize user datasets or provide privacy guarantees. Aggregates can still expose sensitive group names or small cells; rename/remove those fields upstream when an anonymous exported report is required. The synthetic demo's results illustrate behavior only.
