# %% [markdown]
# # Model comparison workbench
#
# All bundled records are generated synthetic examples with neutral labels. Replace the input with your evaluation data and choose any two saved prediction columns. This notebook never fits a model.
#
# Install the notebook extra before starting: `python -m pip install -e ".[notebook]"`.

# %%
# SETUP LOGIC: Synthetic inputs exercise the same API as real evaluation records.
from model_comparison_engine import compare_predictions, show_comparison, Slice
from model_comparison_engine.demo import make_demo

# CONFIGURATION LOGIC: Reproducible demonstration only; these results are not real-world performance evidence.
data = make_demo(n=1200, seed=2026)
predictions = {"Synthetic reference": "reference_prediction", "Synthetic candidate": "candidate_prediction"}

# %% [markdown]
# ## Inspect and compare
#
# Choose model roles, metadata slices, optional numerical bins and two-column interactions. Apply updates the result; exporting preserves the last applied choices. UTC and generic units are defaults, not assumptions about your own data.

# %%
# CONFIGURATION LOGIC: Caller-defined defaults contain no industry-specific fields or business thresholds.
measure = Slice("measure_1", [float("-inf"),0,1,float("inf")], right=False)
# UI LOGIC: The workbench can switch among any supplied prediction columns without inference or training.
panel = show_comparison(data, actual="actual", predictions=predictions,
    id_column="row_id", time_column="time", entity_column="entity_id",
    default_slices=[Slice("segment"), measure], unit="units", timezone="UTC")
# UI LOGIC: Apply the initial declared choices once; subsequent edits require another explicit Apply.
panel.run()

# %% [markdown]
# ## Exact tables and custom exports
#
# Tables stay in variables unless you choose to display them. Exported reports include complete figures, full-precision aggregate tables, support counts and provenance. The two models use the same finite target/prediction records.

# %%
# REPORTING LOGIC: Any pair can be chosen independently of its name or original role.
comparison = compare_predictions(data, "actual", "reference_prediction", "candidate_prediction",
    reference_name="Synthetic reference", candidate_name="Synthetic candidate",
    id_column="row_id", time_column="time", entity_column="entity_id")
# REPORTING LOGIC: These calculations reuse predictions and never fit or infer a model.
summary = comparison.summary()
by_segment = comparison.slice("segment")
interaction = comparison.cross_slice("segment", "category")
# FILE IO LOGIC: Save the computed evidence instead of printing a large table.
report_folder = comparison.export("reports/notebook_demo", slices=["segment",measure],
                                  interactions=[("segment","category")])

# %% [markdown]
# ## Bring your own inputs
#
# Pass a typed DataFrame, CSV or Parquet path to `compare_predictions`. Declare target, prediction columns and any optional identity/time metadata. All additional columns remain available for slices. CSV identity columns are preserved as strings; key types must agree when joining separate prediction tables.
#
# For two fitted estimators, use `compare_models(data, actual=..., reference=Model(...), candidate=Model(...))`. Each `Model` supplies its own ordered features and optional additive offset. See `docs/USAGE.md` for the complete contract.
#
# Generated reports retain your supplied labels. This repository's neutral examples do not automatically anonymize a new dataset.
