# %% [markdown]
# # Model comparison workbench
#
# All bundled records are generated synthetic examples with neutral labels. Replace the input with your evaluation data and choose any two saved prediction columns. This notebook never fits a model.
#
# Install the notebook extra before starting: `python -m pip install -e ".[notebook]"`.

# %%
# SETUP LOGIC: Synthetic inputs exercise the same API as real evaluation records.
from model_comparison_engine import compare_predictions, show_comparison, Slice, InferenceConfig, TemporalConfig
from model_comparison_engine.demo import make_demo
from model_comparison_engine.inference_plots import significance_heatmap
from model_comparison_engine.diagnostic_plots import candidate_figure, worst_slices_figure
from model_comparison_engine.temporal_plots import temporal_figure, event_lag_figure

# CONFIGURATION LOGIC: Reproducible demonstration only; these results are not real-world performance evidence.
data = make_demo(n=3840, seed=2026)
predictions = {"Synthetic reference": "reference_prediction", "Synthetic candidate": "candidate_prediction"}
inference = InferenceConfig(loss="absolute", unit="auto", correction="by", alpha=.05, min_units=10)
# CONFIGURATION LOGIC: Forty synthetic dates allow daily temporal review without asserting a real signal.
timing = TemporalConfig(frequency="1D", signal="bias", rolling_bins=10, max_lag=20)

# %% [markdown]
# ## Inspect and compare
#
# Choose model roles, metadata slices, numerical bins, interactions and statistical settings. Auto uses equally weighted date means when time is configured, otherwise records. Explicit record, date and entity units are available. Negative t favors the candidate; absolute and squared loss are supported. Both minimum records and minimum units must pass.
#
# The v0.4 workbench groups setup into sections, shows a pending-edit badge and wraps long chart labels. Expand the optional time-series controls to change the daily interval, signal and support.
#
# Apply freezes the pair, filters, bins, test settings, candidate population and enabled temporal configuration. Editing controls changes the next Apply; exports preserve the last applied choices. Open Slices → Significance for tests, Candidate for residual plots, and Time series for the enabled timing diagnostics. UTC and generic units are defaults, not assumptions about your own data.

# %%
# CONFIGURATION LOGIC: Caller-defined defaults contain no industry-specific fields or business thresholds.
measure = Slice("measure_1", [float("-inf"),0,1,float("inf")], right=False)
# UI LOGIC: Switch among saved prediction columns and choose all valid candidate records or paired records.
panel = show_comparison(data, actual="actual", predictions=predictions,
    id_column="row_id", time_column="time", entity_column="entity_id",
    default_slices=[Slice("segment"), measure], unit="units", timezone="UTC",
    inference=inference, candidate_population="candidate", temporal=timing)
# UI LOGIC: Apply the initial declared choices once; subsequent edits require another explicit Apply.
panel.run()

# %% [markdown]
# ## Exact paired tables
#
# Paired metrics require finite actual values and both model errors. A paired test on 100 records uses 100 loss differences, not 200 independent observations. For example, reference absolute errors of 2 on every record and candidate absolute errors of 1 on 50 records and 2 on 50 records give a mean paired loss difference of −0.5.
#
# Date/entity tests first average within each unit and then give each unit equal weight. Ninety differences of −1 on one date and ten of +1 on another have record mean −0.8 and date mean 0, with just 2 date units. Ordinary metric tables remain record weighted. Missing unit labels are excluded from tests and disclosed in `tested_n` and `excluded_unit_rows`.

# %%
# REPORTING LOGIC: Any pair can be chosen independently of its name or original role.
comparison = compare_predictions(data, "actual", "reference_prediction", "candidate_prediction",
    reference_name="Synthetic reference", candidate_name="Synthetic candidate",
    id_column="row_id", time_column="time", entity_column="entity_id")
# REPORTING LOGIC: These calculations reuse predictions and never fit or infer a model.
summary = comparison.summary()
by_segment = comparison.slice("segment")
interaction = comparison.cross_slice("segment", "category")
overall_test = comparison.paired_test(inference=inference, min_count=30)
segment_tests = comparison.slice_test("segment", inference=inference, min_count=30)
interaction_tests = comparison.cross_slice_test("segment", "category", inference=inference, min_count=30)
# PLOTTING LOGIC: Keep the complete figure in a variable; display or save it when needed.
significance_plot = significance_heatmap(interaction_tests,
    title="Synthetic candidate vs synthetic reference", unit_label=comparison.unit)

# %% [markdown]
# ## Interpret statistical evidence
#
# The null hypothesis is zero mean candidate-minus-reference loss. Tests are two-sided; negative effects favor the candidate. Squared loss tests mean squared-error differences in squared units, not RMSE differences. Confidence intervals are pointwise and unadjusted.
#
# BY is the default false-discovery-rate correction; BH and no correction are explicit options. Each returned table is its own correction family. Stars use adjusted q thresholds (* ≤ .05, ** ≤ .01, *** ≤ .001); `significant` uses your configured alpha. The † marker denotes low support. Constant or near-constant differences and unsupported cells have no test or stars. Read `status`, `n`, `tested_n` and `unit_count` alongside p/q values.
#
# Dates need not be independent, and out-of-fold predictions can share training data. Date aggregation does not remove serial correlation. BY addresses dependence across valid tests, not dependence within a test. Repeated filters, tables, model choices and exploratory searches are not jointly corrected. No stars, especially with low support, does not establish equivalence. See `docs/USAGE.md` for assumptions and source references.

# %% [markdown]
# ## Candidate residual diagnostics
#
# The default candidate population includes every finite actual/candidate-error record, including missing-reference rows. If 100 candidate records contain 10 missing references, candidate metrics use 100 records while paired metrics use 90. Inspect counts; use `population="paired"` for directly aligned comparison rows. The fully observed synthetic table uses the same records in both populations.
#
# Positive signed residuals mean overprediction. Histograms, calibration, residual-versus-fitted, normal Q–Q, absolute-error and daily plots describe the selected population. Worst slices show support and error contribution; worst cases preserve record identities. These diagnostics do not remove outliers, impute predictions or recalibrate a model.

# %%
# REPORTING LOGIC: Declare the diagnostic population explicitly and retain its support alongside every table.
candidate_population = "candidate"
candidate_rows = comparison.candidate_rows(population=candidate_population)
diagnostics = comparison.candidate_diagnostics(slices=["segment",measure],
    population=candidate_population, min_count=30, top_n=20, bins=10)
paired_diagnostics = comparison.candidate_diagnostics(slices=["segment"], population="paired")
# PLOTTING LOGIC: Calibration levels stay in native target units; residuals use the declared error scale.
candidate_plot = candidate_figure(candidate_rows, diagnostics, name=comparison.candidate_name,
    unit=comparison.unit, error_scale=comparison.config["error_scale"])
weakest_slice_plot = worst_slices_figure(diagnostics["worst_slices"], unit=comparison.unit)

# %% [markdown]
# ## Optional residual timing
#
# Begin with daily bias/MAE, the trailing mean and bin support. Fixed-clock lags preserve elapsed gaps; within-entity event lag one means the next distinct timestamp for that entity, which may be hours or days later. Check median/P90 event gaps before interpreting persistence, and inspect one entity when pooled composition obscures the pattern.
#
# Spectral peaks and offline mean-shift candidates are descriptive, exploratory results. They have no calibrated significance and do not refit a model or repair independence assumptions in the paired tests. Empty bins remain missing. See [Residual timing](docs/TIME_SERIES.md) for the two clocks, support rules and interpretation.

# %%
# REPORTING LOGIC: Existing candidate residuals supply every timing table; all rows retain their original predictions.
timing_tables = comparison.temporal_diagnostics(timing, population=candidate_population)
# REPORTING LOGIC: A fixed-entity view isolates that entity's event gaps and error pattern.
one_entity_timing = comparison.temporal_diagnostics(timing, population=candidate_population, entity="entity_00")
# PLOTTING LOGIC: Elapsed-time bins and within-entity event lags are shown in separate figures.
clock_plot = temporal_figure(timing_tables, name=comparison.candidate_name, unit=comparison.unit)
event_plot = event_lag_figure(timing_tables["event_autocorrelation"], name=comparison.candidate_name)

# %% [markdown]
# ## Reproducible exports
#
# Exact tables remain in variables until displayed. Exported reports contain complete figures, full-precision tables, support counts, inference settings and population provenance. Each export creates a new review directory. The UI export uses its applied state; the API export below uses the explicit configuration in this cell.

# %%
# FILE IO LOGIC: Save the chosen statistical policy and diagnostic population with every review artifact.
report_folder = comparison.export("reports/notebook_demo", slices=["segment",measure],
    interactions=[("segment","category")], min_count=30, inference=inference,
    include_candidate=True, candidate_population=candidate_population, candidate_top_n=20,
    temporal=timing, temporal_population=candidate_population)

# %% [markdown]
# ## Bring your own inputs
#
# Pass a typed DataFrame, CSV or Parquet path to `compare_predictions`. Declare target, prediction columns and optional identity/time metadata. All additional columns remain available for slices. CSV identity columns are preserved as strings; key types must agree when joining separate prediction tables.
#
# For residual predictions, use `reference_offset="known_level"` and/or `candidate_offset="other_known_level"`. Actual 105, residual prediction 2 and offset 100 give reconstructed prediction 102 and error −3 at scale 1. Offsets must be known at prediction time and stay attached to their model when UI roles change.
#
# For two fitted estimators, use `compare_models(data, actual=..., reference=Model(...), candidate=Model(...))`. Each `Model` supplies its own ordered features and optional additive offset. See `docs/USAGE.md` for the complete contract.
#
# Generated reports retain your supplied labels. This repository's neutral examples do not automatically anonymize a new dataset.
