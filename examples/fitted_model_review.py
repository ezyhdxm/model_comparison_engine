# %% [markdown]
# # Fitted models, feature explanations and trade views
#
# This entirely synthetic example demonstrates three already-fitted models, cached pair switching,
# residual-target reconstruction, literal substring filters, held-out explanations and intraday plots.
# The explicit training cell below creates demo models only. The comparison/explanation APIs never fit.
# Install: `python -m pip install -e ".[notebook,interactive]" lightgbm`.
# LightGBM uses native TreeSHAP. For other supported trees, also install `.[explain]`.
#
# For your own evaluation, skip the synthetic generation/training cells and supply your prepared
# evaluation DataFrame plus fitted models. Keep preprocessing and categorical dtypes consistent with training.

# %%
# SETUP LOGIC: Imports do not read private files or train estimators.
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from model_comparison_engine import Model, TradeViewConfig, TemporalConfig, compare_model_set, show_comparison

# %%
# CORE LOGIC: STEP 1 — Generate independent predictors and the synthetic target residual.
# Input: x=1, z=2, noise=0.5, known_level=100.
# Output: residual=0.1, observed=100.1, using residual=2*x-z+0.2*noise.
# Explanation: All model inputs are generated before the target; none contains an observed future outcome.
# Trick: The local random generator makes this demo reproducible without modifying global random state.
rng = np.random.default_rng(2026)
x, z, noise = rng.normal(size=(3, 900))
records = pd.DataFrame({'feature_x':x, 'feature_z':z, 'known_level':100.0})
records['observed'] = records['known_level'] + 2*x - z + 0.2*noise
# CORE LOGIC: STEP 2 — Attach neutral display metadata without changing target values.
# Input: three generated records with row positions=[0,1,2].
# Output: timestamp=['2026-01-05T09:00Z','2026-01-05T09:02Z','2026-01-05T09:04Z'],
# instrument_id=['Asset A','Asset B','Asset C'], issuer=['Group Alpha','Group Beta','Group Gamma'],
# side=['Buy','Sell','Buy'], dealer=['Dealer 1','Dealer 2','Dealer 3'], quantity=[100000,500000,1000000].
# Explanation: All original feature and target values remain on their original records.
# Trick: Metadata are for plotting and slicing; no code guesses trade-side conventions from their values.
records['timestamp'] = pd.date_range('2026-01-05 09:00',periods=len(records),freq='2min',tz='UTC')
records['instrument_id'] = np.resize(['Asset A','Asset B','Asset C'],len(records))
records['issuer'] = np.resize(['Group Alpha','Group Beta','Group Gamma'],len(records))
records['side'] = np.resize(['Buy','Sell'],len(records))
records['dealer'] = np.resize(['Dealer 1','Dealer 2','Dealer 3'],len(records))
records['quantity'] = np.resize([100000,500000,1000000,5000000],len(records))
# CONFIGURATION LOGIC: The full example reserves its last 300 records for evaluation.
training_rows = 600
# CORE LOGIC: STEP 3 — Separate earlier training records from later evaluation records.
# Input: records at positions=[0,1,2], training_rows=2.
# Output: train positions=[0,1], evaluation positions=[2].
# Explanation: No evaluation targets enter the fitting cell below.
# Trick: iloc uses chronological row positions, independent of pandas index labels.
train, evaluation = records.iloc[:training_rows].copy(), records.iloc[training_rows:].copy()

# %%
# CORE LOGIC: STEP 1 — Form the residual training target in the same source units.
# Input: observed=[102,99], known_level=[100,100].
# Output: residual_target=[2,-1].
# Explanation: Prediction reconstruction adds known_level back once for every evaluated model.
# Trick: The offset must be observable at prediction time and is not estimated using evaluation outcomes.
residual_target = train['observed'] - train['known_level']
# MODEL TRAINING LOGIC: Fit only these synthetic demonstration estimators, before entering the engine.
features_a, features_b = ['feature_x'], ['feature_x','feature_z']
model_a = LGBMRegressor(n_estimators=40,num_leaves=7,verbosity=-1,n_jobs=2,random_state=1)
model_b = LGBMRegressor(n_estimators=60,num_leaves=9,verbosity=-1,n_jobs=2,random_state=1)
model_c = LGBMRegressor(n_estimators=20,num_leaves=5,verbosity=-1,n_jobs=2,random_state=1)
model_a.fit(train[features_a],residual_target)
model_b.fit(train[features_b],residual_target)
model_c.fit(train[features_b],residual_target)

# %%
# INFERENCE LOGIC: Cache one prediction per model; any later pair switch reuses those columns.
review = compare_model_set(evaluation,actual='observed',models=[
    Model('Model A',model_a,features_a,offset='known_level'),
    Model('Model B',model_b,features_b,offset='known_level'),
    Model('Model C',model_c,features_b,offset='known_level')],
    reference='Model A',candidate='Model B',time_column='timestamp',entity_column='instrument_id',unit='bps')
# CONFIGURATION LOGIC: Map record attributes explicitly; this example uses already meaningful side labels.
trade_settings = TradeViewConfig(frequency='auto',min_count=5,max_points=300,
    side_column='side',dealer_column='dealer',quantity_column='quantity')
timing = TemporalConfig(frequency='30min',rolling_bins=3,max_lag=5,min_bin_count=5)
# UI LOGIC: Apply, switch any two models, search category values and run explanations through separate buttons.
panel = show_comparison(review,trades=trade_settings,temporal=timing)

# %% [markdown]
# ## Programmatic review without rerunning models
#
# `select_models` changes the selected pair; coverage is recalculated for that pair only.
# A partial string filter matches literal text, not a regular expression.
# Filters compose by intersection and remain attached to exported results.

# %%
# CONFIGURATION LOGIC: These independent review objects reuse cached predictions, with no model inference.
alternative_pair = review.select_models('Model B','Model C')
alpha_slice = review.filter('issuer',values=['alpha'],match='contains',case_sensitive=False)
# REPORTING LOGIC: Full tables remain available without printing them in the notebook.
alpha_metrics = alpha_slice.summary()
trade_tables = review.trade_diagnostics(trade_settings)

# %% [markdown]
# ## Explain the selected models on held-out rows
#
# Positive permutation loss increase means the fitted model relies on that feature. Repeat standard
# deviation describes random shuffles, not a confidence interval. Grouped shuffling preserves the
# relationship between features in a group. A within-context shuffle can be requested through
# `shuffle_within='instrument_id'`; it asks a different, conditional question.
#
# SHAP explains predictions rather than accuracy. The offset is displayed separately from the learned
# contributions. Contributions are checked against the cached model prediction before being returned.
# The UI also supports both applied models and largest-error records from the entire applied population.

# %%
# EXPLANATION LOGIC: Only these explicit calls run extra predictions; the engine never retrains the models.
importance = review.permutation_importance(groups={'Both inputs':features_b},repeats=3,max_rows=200)
candidate_shap = review.shap_values(max_rows=200,max_local_rows=12)
reference_shap = review.shap_values(model='Model A',max_rows=200,max_local_rows=12)

# %%
# FILE IO LOGIC: Export calculated tables, complete PNGs, offline interactive points and cached explanations.
report_folder = review.export('reports/fitted_model_demo',slices=['issuer'],min_count=5,
    trades=trade_settings,temporal=timing)

# %% [markdown]
# The report contains synthetic demonstration results, not evidence about any real dataset.
# In the workbench, use **Export applied review** after requesting UI explanations: that button exports
# the UI's applied comparison. The `review.export` call above exports this notebook variable instead.
# See `docs/USAGE.md`, `docs/EXPLANATIONS.md` and `docs/TRADES.md` for contracts and interpretation.
