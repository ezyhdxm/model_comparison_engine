# Intraday comparisons and individual trade views

These views use existing **paired predictions**: actual, reference and candidate must all be finite after each model's optional offset is added. They never fit a model or request new predictions. Apply population filters first when reviewing an issuer, entity, quantity range or short time interval.

## Two different views

**Paired time comparison** recomputes MAE, RMSE and signed bias using every paired record in each fixed elapsed-time bin. A fourth panel shows record-count bars and the minimum-support threshold. It is useful for finding the hours when one model deteriorates, and distinguishing a persistent bias from changing trade volume. Positive bias means prediction exceeds actual. Low MAE and low bias are different claims: large positive and negative errors can cancel in the bias.

**Paired trade predictions** defaults to four residual panels: candidate and reference signed errors through time, then each model's signed errors against actual values. The latter plots show whether error dispersion grows with spread level. A zero residual means an exact prediction. Optional within-entity and original-level modes show actual and reconstructed predictions, with an identity diagonal in the prediction-versus-actual panels. Every plotted point represents one record; different entities are never joined into a fictitious price path. The interactive version adds zoom and hover details for record ID, entity, time with UTC offset, side, counterparty, dealer, quantity and all three original levels.

Level plots use the original target column's units. Residual and binned error plots use the comparison's `error_scale` and displayed `unit`; the actual-value x-axis below the residual timelines stays in source target units. For example, `actual=1.25` remains `1.25` in a level plot, while a prediction of `1.26` with `error_scale=100` produces a `+1` error in the error display unit. Residual-model offsets are already restored before either view is computed.

## Choose an informative point view

| `point_view` | Display | Use |
|---|---|---|
| `"residual"` (default) | `(prediction - actual) * error_scale`; zero is an exact prediction | Compare errors across instruments without a large range of spread levels making the fit look artificially strong. |
| `"within_entity"` | Subtract the entity's **same mean actual value** from actual and each prediction, then multiply by `error_scale` | Inspect within-instrument deviations after removing each instrument's average level. Requires the comparison's `entity_column`. |
| `"level"` | Original actual and reconstructed prediction values | Review levels, preferably for one focused entity. A tight pooled diagonal can largely reflect differences between bonds. |

Within-entity means use **all focused paired records before point sampling**. For one entity with actual values `[1.0, 1.2]`, candidate predictions `[1.05, 1.1]` and `error_scale=100`, the common actual mean is `1.1`. The centered actual values are `[-10, 10]` and centered predictions are `[-5, 0]`, rounded to numerical display precision. Using the model's own prediction mean instead would hide part of its bias, so the engine does not do that.

Centering removes average level, **not volatility**, and does not demonstrate skill at predicting changes or direction. It uses realized evaluation outcomes and is a retrospective diagnostic, not an available-at-prediction feature. Missing entity IDs produce unavailable centered coordinates; their records remain in the exported source sample and error metrics. A singleton entity has centered actual value zero, which is not evidence of within-entity forecasting skill. Coverage distinguishes sampled records from records with drawable coordinates.

Use `focus_entity` or the notebook's **Focus entity (trade tab only)** selector to inspect one exact source value. `None` means the whole applied paired cohort. The selector includes every nonmissing value in the configured Entity ID column, retaining numeric IDs as numbers. It shows supplied-record counts before the first Apply and last-applied paired counts afterward; no entity is hidden by a top-N cutoff. Missing entity IDs remain available in the all-records view.

Focus changes only the Trade points chart, its interval summaries and their exported artifacts. Other comparison tabs keep the globally applied population. The global population filters still apply first. A missing Entity ID mapping or a focus value absent from the applied paired cohort produces an explicit error. The summary names the actual target column and records both the full applied and focused paired counts.

## Choosing temporal resolution

With `frequency="auto"`, the interval between the earliest and latest local calendar dates determines the resolution:

| Calendar span | Resolution |
|---|---|
| Up to 3 days | 30 minutes |
| 4–14 days | 1 hour |
| More than 14 days | 1 day |

The rule uses calendar span, not only occupied dates. Two trading dates a year apart therefore produce daily bins. You can explicitly choose a fixed interval such as `"5min"`, `"30min"`, `"1h"` or `"1D"`. Calendar-dependent offsets such as months are rejected.

Bins are **left-closed, fixed UTC elapsed intervals anchored to the Unix epoch**. Static plot labels use the comparison's timezone. This preserves distinct instants during a repeated daylight-saving hour. The interactive timeline explicitly displays UTC; its hover also gives the original local timestamp and offset. Daily bins therefore do not necessarily start at local midnight. These plots do not remove nights, weekends or market holidays.

Empty bins have `n=0` and unavailable errors, never zero errors. Bins below `min_count` retain their computed metrics in the exported table but are masked in plotted error curves. This prevents a one-trade bin from looking equally reliable as a thousand-trade bin. Neither binning nor minimum support constitutes a significance test. Unknown-time rows remain in paired coverage and point comparisons but cannot enter the time panels.

An overnight or weekend break is therefore a **calendar gap**, not a sudden improvement to zero error. Read the record-count bars alongside the error panels: an empty-bin gap has no supporting records, while a supported zero MAE or RMSE means the model matched the observations. Zero signed bias can still hide offsetting errors. The notebook summarizes paired records, point sampling, bin support and encodings in wrapping cards; complete summary fields remain in the API and export.

The maximum number of bins is bounded (`max_bins=20000`). Choose a coarser frequency or filter a shorter interval if this bound is exceeded.

## Explicit metadata roles

Map only columns you understand:

- `side_column`: **point color**, preserving literal values. Supplied codes `D`, `B` and `S` consistently use purple, blue and orange. A value `B` is not automatically interpreted as customer buy, dealer buy or bid. Map your prepared column, for example `side_column="side"`; the engine does not construct or reinterpret these labels.
- `dealer_column`: exact dealer identity in interactive hover and the point table; provides category color only when no side column is mapped.
- `counterparty_column`: exact counterparty value in interactive hover and the point table; provides category color only when neither side nor dealer is mapped.
- `quantity_column`: bounded marker area, not an analytical weight. Positive values use `12 + 128 * sqrt(quantity / largest positive sampled quantity)`; unknown, nonfinite, zero and negative values use area 12. The sampled denominator includes retained records without drawable within-entity coordinates. No such records are deleted.

Colors keep up to `max_categories` frequent values (default 12), selected from the complete applied paired population before point sampling, plus separate other and missing categories. Original values stay in hover and the points table. This pooling affects rendering only, not metrics, filters or model features. Labels other than `D`, `B` and `S` receive categorical colors without an invented economic interpretation.

All point panels use the same side colors. **Residual mode uses x markers and a zero-error line**, without an actual-value circle series. In within-entity and level modes, actual values use hollow circles and predictions use x markers. Candidate and reference have separate panels, so model identity does not compete with side for color. Dealer and counterparty remain available in hover even when they do not control color. Long labels wrap, and the legend states the applied color mapping.

## Sampling and reproducibility

The point view displays at most `max_points=2000` records using a seeded, uniform sample without replacement. The default seed is `random_state=0`. The displayed fraction is disclosed. Rare extreme events may be absent from this overview; use the existing worst-case table or filter the relevant entity to inspect those events. The point sample never changes full-population bin metrics or inference.

```python
# SETUP LOGIC: Import optional trade-review functions.
from model_comparison_engine.trade_view import TradeViewConfig, trade_tables
from model_comparison_engine.trade_plots import (
    intraday_figure, trade_figure, trade_interactive,
)

# CONFIGURATION LOGIC: Use your actual column names; optional roles may be None.
settings = TradeViewConfig(
    frequency="auto", min_count=30, max_points=2000,
    side_column="side", counterparty_column="counterparty_type",
    dealer_column="dealer_id", quantity_column="quantity",
    point_view="residual", focus_entity=None,
)

# DIAGNOSTIC LOGIC: Reuse the existing Comparison and its applied paired population.
tables = trade_tables(comparison, settings)

# PLOTTING LOGIC: Static figures are available without Plotly.
time_figure = intraday_figure(tables, unit=comparison.unit)
points_figure = trade_figure(tables)

# UI LOGIC: Optional hover and zoom; install the interactive extra first.
interactive_figure = trade_interactive(tables)
interactive_figure.show()
```

Install optional interactivity with `pip install "model-comparison-engine[interactive]"`. Static figures and tables continue to work without this extra.

## Reading residual time diagnostics

The separate temporal reading guide connects every observation to an interpretation and next check:

- **Population/clock:** inspect gaps, missing timestamps and supported bins before drawing conclusions.
- **Residual signal:** positive signed bias means overprediction; compare it with MAE and slices because cancellation can hide large errors.
- **Fixed-clock correlation:** inspect both correlation and pair count. A scanned maximum is exploratory and is not a significance result. Entity composition can create apparent persistence.
- **Lomb–Scargle spectrum:** a peak describes how well a sinusoid fits the selected binned signal. It is not a p-value or prediction improvement. Compare the sampling-window curve and weekday/hour coverage; trading calendars can create aliases. Confirm a proposed cycle in another interval before making a modeling decision.
- **Mean-shift candidates:** a heuristic offline scan highlights windows to investigate, not causal events or exact break times. Zero candidates does not establish stability.
- **Within-entity event lags:** lag 1 means the next distinct timestamp within the selected entity's evaluation records. It need not be the next market trade. Equal-time rows are averaged; pooled correlations are not entity-demeaned, so persistent entity bias and active entities can dominate.

For numeric observations and this guide together, call `temporal_interpretation(comparison.temporal_diagnostics(), unit=comparison.unit)` from `model_comparison_engine.temporal_interpretation`. Temporal diagnostics and paired intraday model metrics answer different questions: the former describes one selected residual signal; the latter compares both models on the same records through time.
