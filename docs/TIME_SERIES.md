# Residual timing: what to inspect and why

The comparison engine evaluates scalar predictions on observed events. Event timestamps need not be equally spaced, and many instruments can trade at the same time. A transaction-ordered residual array is therefore not automatically a regularly sampled time series.

This release adds exploratory temporal diagnostics for existing candidate predictions. They answer where a model's errors persist or change. They do not establish profitable predictability, estimate a quote arrival process, or refit a model. Use evaluation predictions with the same target reconstruction and error units as the main comparison.

## Recommended order

| Diagnostic | Question | Useful follow-up |
|---|---|---|
| Binned bias, MAE, trailing mean and support | Does the model become persistently high, low, or less accurate? | Check timestamp mapping, source changes, benchmark adjustments and evaluation-period composition. |
| Within-entity event-lag correlation | Does an instrument's previous prediction error resemble its next error? | Review a fixed entity or cohort and investigate prediction-time history features. Validate any proposed feature separately. |
| Fixed-clock lag correlation | Do errors at a given elapsed interval move together? | Investigate calendar/session effects and whether independence assumptions behind loss tests are plausible. |
| Floating-mean Lomb–Scargle spectrum and sampling window | Is a recurring component visible in the supported binned residuals? | Check whether the peak follows the observation calendar, changing cohorts or a trend before attributing it to a missing feature. |
| Offline mean-shift candidates | Which time brackets separate different average bias or MAE? | Inspect data and model changes around the bracket. Compare the same maturity, size and entity mix on both sides. |

The first three checks usually have the clearest interpretation for spread prediction. A spectral peak is supporting evidence, not a feature-selection criterion. Change candidates localize a review; they do not justify deleting an inconvenient period.

## Two clocks

**Clock bins** use fixed elapsed UTC durations. `frequency="1D"` means 24 hours, and `"1h"` means 3,600 seconds. These are not exchange-session calendars or local days around daylight-saving changes. Bin boundaries are left closed, right open, and anchored to the Unix epoch. All bins from the first through last timed record are retained. Empty bins have `n=0` and missing residual means; they are never filled with zero or interpolated.

Within each occupied bin, bias and MAE are record-weighted averages. Subsequent rolling means, spectra and mean-shift searches give each supported bin equal weight. `min_bin_count` controls support. The trailing window includes its current bin, spans `rolling_bins` elapsed bins and averages the available supported signals; `rolling_supported_bins` shows exactly how many contributed. Sparse windows can therefore have fewer observations than the declared width. This is retrospective diagnosis, not a causal feature construction routine.

**Event lags** pair distinct prediction timestamps within the same declared entity. Simultaneous predictions for an entity are averaged for this diagnostic only, preventing input row order from inventing a lead/lag. Lag one means the previous distinct timestamp for that entity, even if another entity has traded in between. Median and P90 actual time gaps accompany every lag. Missing identity/time and collapsed ties are counted. Unknown identifiers are not combined into a fabricated entity.

Event correlations pool same-entity pairs and weight active entities more heavily. They can include persistent entity-level bias as well as serial dependence. A fixed-entity review is often easier to interpret. Neither clock correlation table provides an IID confidence band or a whiteness-test p-value.

## Spectrum

[VanderPlas (2018)](https://arxiv.org/abs/1703.09824) explains why uneven sampling and observation windows matter when interpreting Lomb–Scargle peaks. The engine uses [SciPy's generalized, floating-mean Lomb–Scargle](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html), supported by SciPy 1.15 or later, on the actual UTC centers of supported bins. Removing unsupported bins never compresses their elapsed time gaps.

The scan uses 512 trial frequencies, normalized power and equal bin weights. The floating mean is fitted independently at each frequency. By default, the shortest period is four bin widths and the longest is half the supported time span, allowing at least two cycles. At least 20 supported bins are required. `period_min` and `period_max` are optional seconds; the upper bound is always limited by half the supported span. Explicit short-period choices can still be aliased or unresolved by the binned observations. Bin aggregation can hide faster effects.

The sampling window is the squared magnitude of the mean observation-time phasor at each trial frequency. It describes cadence, not residual power or a probability. Similar peaks merit checking trading hours, missing days and coverage. Irregular sampling is not an automatic remedy for aliasing. Constant signals, insufficient bins or unresolved period ranges produce an explicit unavailable status. The largest scanned peak has no automatic significance claim: serially correlated residuals, trends, multiple-frequency searching and nonstationarity matter.

## Change candidates

[Truong, Oudre and Vayatis (2020)](https://arxiv.org/abs/1801.00718) organize offline detection around the segment cost, search procedure and complexity constraint. This implementation uses a deliberately inspectable combination:

- The signal is either supported-bin mean signed residual (`bias`) or mean absolute error (`mae`). A change in MAE concerns error magnitude, not a formal variance-change test.
- The cost is within-segment squared error about its mean. A centered, bounded normalization avoids arithmetic overflow.
- Greedy binary segmentation selects the largest admissible cost reduction, then repeats within the resulting segments. It is not an exact globally optimal segmentation or PELT implementation.
- Each child requires `min_segment_bins` actual supported bins. The default is eight, with at most three selected changes.
- A reduction must exceed `change_penalty * variance(normalized_signal) * log(number_of_supported_bins)`. The default multiplier is three. This is a descriptive tuning rule, not a p-value, BIC claim, or calibrated alarm rate.

The table preserves selection rank, before/after means, supported-bin counts and the last observed bin center before/first after the boundary. A gap leaves a bracket, not a precisely observed change time. The plotted midpoint is only a visual marker. Means refer to the local parent segment when the candidate was selected; later subdivisions do not rewrite that historical comparison. Outliers, changes in trade mix, or a poorly chosen penalty can create or conceal candidates. A no-candidate result is not evidence of stationarity.

We do not automatically apply the [Ljung–Box test](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html) to an interleaved, irregular trade array. Its lag and degrees-of-freedom interpretation requires a defensible series and model specification. We also do not turn the candidate search into an online CUSUM alarm: a calibrated [CUSUM](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm) needs an appropriate baseline and control-limit design. Strong temporal dependence should motivate a separate inference design, such as an appropriate block procedure, rather than interpreting the existing independent-unit t-test more confidently.

## Use in a notebook

In `show_comparison`, expand **Time-series diagnostics**, enable the option, select frequency, signal and support, and click **Apply comparison**. Open **Time series**. Population filters and the candidate-population choice also apply to these diagnostics. To inspect a single instrument in the UI, filter its mapped entity column. Export records the applied settings, not pending controls.

```python
# SETUP LOGIC: Diagnostics operate on existing comparison predictions.
from model_comparison_engine import TemporalConfig
from model_comparison_engine.temporal_plots import temporal_figure, event_lag_figure

# CONFIGURATION LOGIC: All thresholds and time units are explicit and remain fixed within a review.
timing = TemporalConfig(frequency="1D", signal="bias", rolling_bins=10,
    max_lag=20, min_bin_count=10, min_segment_bins=8,
    change_penalty=3.0, max_changes=3)

# REPORTING LOGIC: Choose the independently valid candidate population or the common paired population.
timing_tables = comparison.temporal_diagnostics(timing, population="candidate")
one_entity = comparison.temporal_diagnostics(timing, population="paired", entity="instrument_001")

# PLOTTING LOGIC: The clock series and within-entity event lags describe different notions of adjacency.
clock_figure = temporal_figure(timing_tables, name=comparison.candidate_name, unit=comparison.unit)
event_figure = event_lag_figure(timing_tables["event_autocorrelation"], name=comparison.candidate_name)

# FILE IO LOGIC: Temporal tables and full figures join the ordinary paired review.
folder = comparison.export("reports/review", slices=["sector"], temporal=timing,
    temporal_population="candidate", temporal_entity=None)
```

`settings=None` in `temporal_diagnostics` uses `TemporalConfig()` defaults. In `export`, `temporal=None` omits the optional temporal section. Walk-forward result exports accept the same temporal arguments. All parameters can also be supplied in the CLI configuration's `temporal` object. The default 20,000-bin bound prevents a short-frequency setting on a long history from allocating an uncontrolled grid; choose a coarser frequency or a narrower population when prompted.

The returned tables are `summary`, `series`, `autocorrelation`, `spectrum`, `change_points`, and `event_autocorrelation`. Their CSV values are computed directly from the supplied predictions. Time-unavailable, low-support, constant-signal and no-candidate cases remain explicit. For residual-target models, reconstruction offsets are already applied by the comparison layer before these diagnostics run.
