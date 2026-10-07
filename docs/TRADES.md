# Intraday comparisons and individual trade views

These views use existing **paired predictions**: actual, reference and candidate must all be finite after each model's optional offset is added. They never fit a model or request new predictions. Apply population filters first when reviewing an issuer, entity, quantity range or short time interval.

## Two different views

**Paired time comparison** recomputes MAE, RMSE and signed bias using every paired record in each fixed elapsed-time bin. A fourth panel shows record counts. It is useful for finding the hours when one model deteriorates, and distinguishing a persistent bias from changing trade volume. Positive bias means prediction exceeds actual. Low MAE and low bias are different claims: large positive and negative errors can cancel in the bias.

**Paired trade predictions** shows actual and both reconstructed predictions through time, plus predictions versus actual. Every dot represents one record; different entities are never joined into a fictitious price path. The diagonal in the prediction-versus-actual panel means exact prediction. The interactive version adds zoom and hover details for record ID, entity, time with UTC offset, side, counterparty, dealer, quantity and all three levels.

Level plots use the original target column's units. Binned error plots use the comparison's `error_scale` and displayed `unit`. For example, `actual=1.25` remains `1.25` in a level plot, while a prediction of `1.26` with `error_scale=100` produces a `+1` error in the error display unit. Residual-model offsets are already restored before either view is computed.

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

The maximum number of bins is bounded (`max_bins=20000`). Choose a coarser frequency or filter a shorter interval if this bound is exceeded.

## Explicit metadata roles

Map only columns you understand:

- `side_column`: marker shape, preserving literal values. A value `B` is not automatically interpreted as customer buy, dealer buy or bid.
- `dealer_column`: candidate-point color when supplied.
- `counterparty_column`: color when dealer is not supplied; always available in interactive hover when mapped.
- `quantity_column`: bounded marker area, not an analytical weight. Positive values use `12 + 128 * sqrt(quantity / largest positive plotted quantity)`; unknown, nonfinite, zero and negative values use area 12. No such records are deleted.

The largest five side categories receive individual shapes. Other sides share an explicit display-only category. Colors keep up to `max_categories` frequent plotted values (default 12), plus separate other and missing categories. Original values stay in hover and the points table. This pooling affects rendering only, not metrics, filters or model features.

On the static timeline, colors distinguish actual/reference/candidate and symbols show side. On the candidate-versus-actual panel, colors distinguish dealer/counterparty and symbols still show side. The reference appears as faint `+` markers for context. Long labels wrap and figure height increases with the legend.

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
