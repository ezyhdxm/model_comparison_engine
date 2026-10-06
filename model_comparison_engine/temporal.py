"""Exploratory residual timing: UTC bins, gap-aware correlation and mean-shift candidates.

Lomb–Scargle uses a floating mean on occupied bin centers, with equal bin weights.
Its normalized power and the binary-segmentation penalty are descriptive quantities,
not significance tests. No missing observation is replaced with a zero residual.
"""
# SETUP LOGIC: Only already-prepared evaluation errors enter these diagnostics.
from dataclasses import dataclass
from numbers import Integral, Real
import numpy as np
import pandas as pd
from scipy.signal import lombscargle

# CONFIGURATION LOGIC: Stable schemas also describe unavailable diagnostics.
SERIES_COLUMNS = ['time', 'time_center', 'n', 'entities', 'bias', 'mae', 'supported',
                  'signal', 'rolling_signal', 'rolling_supported_bins']
ACF_COLUMNS = ['lag_bins', 'lag_seconds', 'pair_n', 'correlation']
SPECTRUM_COLUMNS = ['frequency_hz', 'period_seconds', 'power', 'sampling_window']
CHANGE_COLUMNS = ['rank', 'left_time', 'right_time', 'boundary_time', 'left_n', 'right_n',
                  'mean_before', 'mean_after', 'reduction_normalized', 'penalty_normalized']
SPECTRUM_MIN_BINS = 20
SPECTRUM_GRID_SIZE = 512
ACF_MIN_PAIRS = 3


@dataclass(frozen=True)
class TemporalConfig:
    """Fixed elapsed-time bins; optional spectrum period bounds are in seconds."""
    # CONFIGURATION LOGIC: Daily UTC bins provide a conservative default for irregular trades.
    frequency: str = '1D'
    signal: str = 'bias'
    rolling_bins: int = 10
    max_lag: int = 20
    min_bin_count: int = 1
    min_segment_bins: int = 8
    change_penalty: float = 3.0
    max_changes: int = 3
    period_min: float | None = None
    period_max: float | None = None
    max_bins: int = 20000

    def __post_init__(self):
        # VALIDATION LOGIC: Reject calendar-dependent offsets and unsafe allocation parameters early.
        _bin_nanoseconds(self.frequency)
        if self.signal not in {'bias', 'mae'}:
            raise ValueError("Temporal signal must be 'bias' or 'mae'.")
        for name in ('rolling_bins', 'min_bin_count', 'min_segment_bins', 'max_bins'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        for name in ('max_lag', 'max_changes'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer.')
        if self.max_lag > 1000 or self.max_changes > 100:
            raise ValueError('Use max_lag <= 1000 and max_changes <= 100 to bound the temporal scans.')
        if not isinstance(self.change_penalty, Real) or not np.isfinite(self.change_penalty) or self.change_penalty < 0:
            raise ValueError('change_penalty must be finite and nonnegative.')
        for name in ('period_min', 'period_max'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, Real) or not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be positive finite seconds.')
        if self.period_min is not None and self.period_max is not None and self.period_min >= self.period_max:
            raise ValueError('period_min must be smaller than period_max.')


def _bin_nanoseconds(frequency):
    # VALIDATION LOGIC: UTC removes DST ambiguity, but months and business days still lack fixed durations.
    try:
        nanos = pd.tseries.frequencies.to_offset(frequency).nanos
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Use a fixed temporal frequency, such as '1h' or '1D'; calendar offsets are unsupported.") from exc
    if nanos <= 0:
        raise ValueError('Temporal frequency must have a positive duration.')
    return int(nanos)


def _time_values(rows):
    # TIME CONVERSION LOGIC: Prepared missing timestamps retain coverage; known instants share one UTC clock.
    if '__time' not in rows:
        return pd.Series(pd.NaT, index=rows.index, dtype='datetime64[ns, UTC]')
    supplied = rows['__time']
    if supplied.notna().any() and getattr(supplied.dtype, 'tz', None) is None:
        raise ValueError('Prepared temporal rows require timezone-aware __time values.')
    return pd.to_datetime(supplied, utc=True).dt.as_unit('ns')


def _bounded_grid(times, config):
    # CORE LOGIC: STEP 1 — Determine the complete fixed-width span before allocating any empty bins.
    # Input: UTC times=['2026-01-01 02:00','2026-01-03 00:00'], frequency='1D', max_bins=20000.
    # Output: first=2026-01-01 00:00 UTC, count=3, bin_ns=86400000000000.
    # Explanation: The exact midnight record belongs to January 3, so the empty January 2 bin is retained.
    # Trick: Python integers prevent span arithmetic overflow; an epoch anchor also makes repeated calls align.
    bin_ns = _bin_nanoseconds(config.frequency)
    first = (int(times.min().value) // bin_ns) * bin_ns
    last = (int(times.max().value) // bin_ns) * bin_ns
    count = (last - first) // bin_ns + 1
    # VALIDATION LOGIC: Bound memory before constructing a date range or resampling sparse long histories.
    if count > config.max_bins:
        raise ValueError(f'Temporal span needs {count:,} bins, exceeding max_bins={config.max_bins:,}; '
                         'choose a coarser frequency or a shorter date range.')
    if first < pd.Timestamp.min.value or last + bin_ns // 2 > pd.Timestamp.max.value:
        raise ValueError('Temporal bin boundaries exceed the supported timestamp range; shorten the date range.')
    # CORE LOGIC: STEP 2 — Build bins at equal elapsed UTC durations, including daylight-saving transitions.
    # Input: first=2026-03-08 00:00 UTC, count=3, bin_ns=86400000000000.
    # Output: grid=[2026-03-08 00:00,2026-03-09 00:00,2026-03-10 00:00] UTC.
    # Explanation: Every adjacent pair is exactly 86400 seconds apart even when local civil clocks change.
    # Trick: Integer multiplication happens with Python integers before datetime conversion to avoid int64 wrap.
    grid = pd.to_datetime([first + index * bin_ns for index in range(count)], utc=True)
    return grid, bin_ns


def _series(rows, times, config):
    # CORE LOGIC: STEP 1 — Place each timed residual into its left-closed UTC bin using positional alignment.
    # Input: times=['2026-01-01 01:00','2026-01-01 23:00','2026-01-03 00:00'] UTC, errors=[-1,3,2].
    # Output: bin_positions=[0,0,2], grid starts=[2026-01-01,2026-01-02,2026-01-03] UTC.
    # Explanation: Two January 1 records share a bin; January 2 remains an explicit empty position.
    # Trick: Positional arrays prevent duplicate source indices from changing alignment.
    timed = rows.loc[times.notna()]
    available = times.loc[times.notna()]
    grid, bin_ns = _bounded_grid(available, config)
    positions = np.array([(int(value) - grid[0].value) // bin_ns for value in available.astype('int64')])
    # CORE LOGIC: STEP 2 — Normalize each occupied bin independently before averaging its errors.
    # Input: bin_positions=[0,0,2], errors=[-1,3,2], absolute_errors=[1,3,2], entities=['A','B','A'].
    # Output: per-record scales=[3,3,2], scaled errors=[-1/3,1,1], scaled absolute errors=[1/3,1,1].
    # Explanation: January 3 uses its own scale 2 rather than the scale 3 in January 1.
    # Trick: Per-bin scaling avoids both overflowing a bin mean and underflowing tiny errors in unrelated bins.
    observations = pd.DataFrame({'bin': positions, 'error': timed['__error_candidate'].to_numpy(),
                                  'absolute': timed['__ae_candidate'].to_numpy(),
                                  'entity': timed['__entity'].to_numpy()})
    observations['scale'] = observations.groupby('bin', sort=False)['absolute'].transform('max')
    divisor = observations['scale'].where(observations['scale'].gt(0), 1.0)
    observations[['error', 'absolute']] = observations[['error', 'absolute']].div(divisor, axis=0)
    grouped = observations.groupby('bin', sort=True)
    # CORE LOGIC: STEP 3 — Aggregate every bin while keeping absent residual means undefined.
    # Input: bin_positions=[0,0,2], errors=[-1,3,2], entities=['A','B','A'], grid length=3.
    # Output: n=[2,0,1], entities=[2,0,1], bias=[1,NaN,2], mae=[2,NaN,2].
    # Explanation: Both records contribute to the first bin's averages; an empty bin contributes no zero error.
    # Trick: Row-aligned multiplication restores each bin's units; reindex adds gaps after occupied-bin aggregation.
    aggregated = grouped.agg(n=('error', 'size'), entities=('entity', 'nunique'),
                             bias=('error', 'mean'), mae=('absolute', 'mean'),
                             scale=('scale', 'first')).reindex(range(len(grid)))
    aggregated[['n', 'entities']] = aggregated[['n', 'entities']].fillna(0).astype('int64')
    aggregated[['bias', 'mae']] = aggregated[['bias', 'mae']].mul(aggregated['scale'], axis=0)
    aggregated.insert(0, 'time_center', grid + pd.Timedelta(bin_ns // 2, unit='ns'))
    aggregated.insert(0, 'time', grid)
    # CORE LOGIC: STEP 4 — Mask weak bins and calculate a trailing mean with explicit observed-bin support.
    # Input: bias=[1,NaN,2], n=[2,0,1], signal='bias', min_bin_count=1, rolling_bins=2.
    # Output: supported=[True,False,True], signal=[1,NaN,2], rolling_signal=[1,1,2], rolling_supported_bins=[1,1,1].
    # Explanation: Each trailing two-bin window averages its available supported signals with equal bin weights.
    # Trick: Window width counts elapsed bins, not successive observations; the support count exposes sparse windows.
    aggregated['supported'] = aggregated['n'].ge(config.min_bin_count)
    aggregated['signal'] = aggregated[config.signal].where(aggregated['supported'])
    aggregated['rolling_signal'] = _rolling_signal(aggregated['signal'], config.rolling_bins)
    aggregated['rolling_supported_bins'] = aggregated['signal'].rolling(config.rolling_bins, min_periods=1).count().astype('int64')
    return aggregated[SERIES_COLUMNS].reset_index(drop=True)


def _rolling_signal(signal, window):
    # CORE LOGIC: STEP 1 — Detect whether one global scale would underflow any finite, nonzero bin signal.
    # Input: signal=[1e308,1e-20,2e-20], window=2.
    # Output: scale=1e308, normalized=[1,0,0], needs_local=True, effective window=2.
    # Explanation: The small errors divided by 1e308 round to zero, so they require a window-specific scale.
    # Trick: Subnormal normalized values also select the guarded path; all-missing input keeps a missing rolling curve.
    scale = float(signal.abs().max())
    normalized = signal / scale if scale > 0 else signal.copy()
    nonzero = signal.notna() & signal.ne(0)
    needs_local = normalized.loc[nonzero].abs().lt(np.finfo(float).tiny).any()
    window = min(window, len(signal))
    # CORE LOGIC: STEP 2 — Average observed values in each trailing window without borrowing a distant bin's scale.
    # Input: signal=[1e308,1e-20,2e-20], window=2, needs_local=True.
    # Output: rolling_signal=[1e308,5e307,1.5e-20] (floating-point approximation).
    # Explanation: Once the large first value leaves the window, the two small values retain their own mean.
    # Trick: Ordinary ranges use a linear rolling mean; only extreme ranges use bounded per-window stable means.
    if needs_local:
        return signal.rolling(window, min_periods=1).apply(_finite_window_mean, raw=True)
    restored_scale = scale if scale > 0 else 1.0
    return normalized.rolling(window, min_periods=1).mean() * restored_scale


def _finite_window_mean(values):
    # CORE LOGIC: STEP 1 — Remove only missing bin signals before a window-local stable average.
    # Input: values=[1e-20,NaN,2e-20].
    # Output: 1.5e-20 (floating-point approximation).
    # Explanation: The two observed values average with equal weights; the missing bin contributes no zero.
    # Trick: pandas min_periods=1 guarantees at least one finite value before this callback is invoked.
    return _stable_mean(values[np.isfinite(values)])


def _centered_scale(values):
    # CORE LOGIC: STEP 1 — Center and normalize finite values before correlations or squared-error arithmetic.
    # Input: values=[0,0,4,4].
    # Output: [-1,-1,1,1].
    # Explanation: Dividing by 4, subtracting .5, then dividing by .5 produces a bounded zero-mean signal.
    # Trick: Scaling before centering avoids overflow for opposite extreme finite errors; constants map to zero.
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values))) if len(values) else 0.0
    normalized = values / scale if scale > 0 else values.copy()
    centered = normalized - np.mean(normalized) if len(values) else normalized
    radius = float(np.max(np.abs(centered))) if len(centered) else 0.0
    return centered / radius if radius > 0 else centered


def _pair_correlation(left, right):
    # CORE LOGIC: STEP 1 — Compute Pearson correlation over an already-aligned, finite pair population.
    # Input: left=[1,2,3], right=[2,4,6].
    # Output: 1.0.
    # Explanation: Both centered normalized arrays are [-1,0,1], so covariance equals both standard deviations.
    # Trick: At least three pairs and nonconstant values on both sides are required; otherwise return NaN.
    if len(left) < ACF_MIN_PAIRS:
        return np.nan
    x, y = _centered_scale(left), _centered_scale(right)
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.clip(np.dot(x, y) / denominator, -1, 1)) if denominator > 0 else np.nan


def _autocorrelation(series, config):
    # CORE LOGIC: STEP 1 — Match observations separated by actual fixed-grid lags without collapsing gaps.
    # Input: signal=[1,NaN,2,3,4,5], bin_seconds=86400, max_lag=1.
    # Output: lag_bins=1, lag_seconds=86400, pair_n=3, correlation=1.0 from pairs [(2,3),(3,4),(4,5)].
    # Explanation: January 1 is not paired with January 3 because that separation spans two bins.
    # Trick: Finite masks apply after lagging; missing bins remain on the elapsed-time clock.
    values = series['signal'].to_numpy(dtype=float)
    bin_seconds = _bin_nanoseconds(config.frequency) / 1e9
    records = []
    for lag in range(1, min(config.max_lag, max(0, len(values) - 1)) + 1):
        left, right = values[:-lag], values[lag:]
        finite = np.isfinite(left) & np.isfinite(right)
        records.append(dict(lag_bins=lag, lag_seconds=lag * bin_seconds, pair_n=int(finite.sum()),
                            correlation=_pair_correlation(left[finite], right[finite])))
    return pd.DataFrame(records, columns=ACF_COLUMNS)


def _period_range(observed, config):
    # CORE LOGIC: STEP 1 — Choose default bin-scale periods and require at least two observed cycles.
    # Input: observed centers span=100 days, frequency='1D', period_min=None, period_max=None.
    # Output: period_min_seconds=345600, period_max_seconds=4320000 (4 days to 50 days).
    # Explanation: Four bins per shortest default cycle and two cycles across the span reduce edge extrapolation.
    # Trick: Explicit bounds use seconds; even an explicit upper bound is capped at half the supported time span.
    bin_seconds = _bin_nanoseconds(config.frequency) / 1e9
    span = (int(observed['time_center'].iloc[-1].value) - int(observed['time_center'].iloc[0].value)) / 1e9 if len(observed) else 0.0
    lower = float(config.period_min) if config.period_min is not None else 4 * bin_seconds
    upper = min(float(config.period_max), span / 2) if config.period_max is not None else span / 2
    return lower, upper


def _spectrum(series, config):
    # CORE LOGIC: STEP 1 — Select supported bins while retaining their original elapsed time separations.
    # Input: supported centers=[2026-01-01 12:00,2026-01-03 12:00,2026-01-04 12:00] UTC, signal=[0,1,0].
    # Output: the three supported centers are retained, default period bounds=[345600,129600] seconds,
    #         status='insufficient_bins' because 3 < 20.
    # Explanation: The missing January 2 bin creates a two-day first gap, not a shortened observation sequence.
    # Trick: Equal bin weights describe the selected binned signal; heavily traded bins do not automatically dominate.
    observed = series.loc[series['supported']].reset_index(drop=True)
    lower, upper = _period_range(observed, config)
    context = dict(spectrum_min_bins=SPECTRUM_MIN_BINS, period_min_seconds=lower,
                   period_max_seconds=upper, spectrum_peak_period_seconds=np.nan, spectrum_peak_power=np.nan)
    empty = pd.DataFrame(columns=SPECTRUM_COLUMNS)
    if len(observed) < SPECTRUM_MIN_BINS:
        return empty, dict(context, spectrum_status='insufficient_bins')
    # VALIDATION LOGIC: A constant series or unresolved frequency interval has no interpretable spectrum.
    normalized = _centered_scale(observed['signal'].to_numpy(dtype=float))
    if not np.any(normalized):
        return empty, dict(context, spectrum_status='constant_signal')
    if upper <= lower:
        return empty, dict(context, spectrum_status='inadequate_period_range')
    # CORE LOGIC: STEP 2 — Fit floating-mean sinusoids on the irregular observed-bin times and inspect cadence aliases.
    # Input: elapsed_seconds=[0,1,3,4], normalized=[0,1,-1,0], angular_frequency=2*pi/4.
    # Output: fitted power=1.0 and sampling_window=0.25 at period=4 seconds (approximately).
    # Explanation: These four residuals equal sin(pi*t/2); the sampling phasors [1,i,-i,1] average to .5.
    # Trick: Lomb–Scargle receives radians/second, not Hz; the floating mean is re-fitted at every trial frequency.
    origin = int(observed['time_center'].iloc[0].value)
    elapsed = np.array([(int(value.value) - origin) / 1e9 for value in observed['time_center']])
    frequencies = np.linspace(1 / upper, 1 / lower, SPECTRUM_GRID_SIZE)
    angular = 2 * np.pi * frequencies
    power = lombscargle(elapsed, normalized, angular, normalize=True, floating_mean=True)
    window = np.array([abs(np.exp(1j * frequency * elapsed).mean()) ** 2 for frequency in angular])
    result = pd.DataFrame(dict(frequency_hz=frequencies, period_seconds=1 / frequencies,
                               power=power, sampling_window=window), columns=SPECTRUM_COLUMNS)
    # CORE LOGIC: STEP 3 — Summarize the largest scanned power without assigning statistical significance.
    # Input: periods_seconds=[8,4,2], power=[.1,.8,.2].
    # Output: spectrum_peak_period_seconds=4, spectrum_peak_power=.8, spectrum_status='ok'.
    # Explanation: The strongest trial frequency is a descriptive scan result, not a confirmed periodic process.
    # Trick: The 512-point frequency grid limits peak resolution; correlated errors and cadence can create apparent peaks.
    if not np.isfinite(power).all():
        return empty, dict(context, spectrum_status='numerically_unavailable')
    peak = int(np.argmax(power))
    context.update(spectrum_peak_period_seconds=float(result['period_seconds'].iloc[peak]),
                   spectrum_peak_power=float(power[peak]), spectrum_status='ok')
    return result, context


def _best_split(prefix, start, end, minimum):
    # CORE LOGIC: STEP 1 — Enumerate only boundaries with enough actual observations in both child segments.
    # Input: normalized=[-1,-1,1,1], prefix=[0,-1,-2,-1,0], start=0, end=4, minimum=2.
    # Output: positions=[2], left_n=[2], right_n=[2], left_sum=[-2], right_sum=[2].
    # Explanation: Boundary 2 leaves exactly two supported bins on either side; no missing bin is treated as a value.
    # Trick: Segment bounds use half-open indexing, and the last admissible split end-minimum is included.
    if end - start < 2 * minimum:
        return None
    positions = np.arange(start + minimum, end - minimum + 1)
    left_n, right_n = positions - start, end - positions
    left_sum = prefix[positions] - prefix[start]
    right_sum = prefix[end] - prefix[positions]
    # CORE LOGIC: STEP 2 — Evaluate the reduction in within-segment squared error using bounded prefix sums.
    # Input: left_sum=[-2], right_sum=[2], left_n=[2], right_n=[2], parent_sum=0, parent_n=4.
    # Output: best split=2, reduction_normalized=4.0.
    # Explanation: The parent SSE is 4 and both constant children have SSE 0, giving an improvement of 4.
    # Trick: Squared-observation terms cancel; the nonnegative clamp absorbs only floating-point cancellation at zero.
    parent_term = (prefix[end] - prefix[start]) ** 2 / (end - start)
    reductions = np.maximum(left_sum ** 2 / left_n + right_sum ** 2 / right_n - parent_term, 0)
    best = int(np.argmax(reductions))
    return int(positions[best]), float(reductions[best]), start, end


def _stable_mean(values):
    # CORE LOGIC: STEP 1 — Restore a mean in residual units without overflowing a finite segment average.
    # Input: values=[-1e308,1e308].
    # Output: 0.0.
    # Explanation: The normalized values [-1,1] average to zero before multiplication by 1e308.
    # Trick: Zero-only segments return zero directly; callers always supply at least one finite observation.
    scale = float(np.max(np.abs(values)))
    return float(np.mean(values / scale) * scale) if scale > 0 else 0.0


def _change_record(observed, candidate, rank, penalty):
    # CORE LOGIC: STEP 1 — Preserve the observed time bracket and both local segment means for a selected split.
    # Input: centers=[Jan 1 noon,Jan 2 noon,Jan 5 noon,Jan 6 noon] UTC, signal=[0,0,4,4],
    #        candidate=(split=2,reduction=4,start=0,end=4), rank=1, penalty=1.
    # Output: rank=1, left_time=Jan 2 noon, right_time=Jan 5 noon, boundary_time=Jan 4 midnight, left_n=2, right_n=2,
    #         mean_before=0, mean_after=4, reduction_normalized=4, penalty_normalized=1.
    # Explanation: No trade-supported bin locates the shift more precisely than the three-day observed bracket.
    # Trick: The midpoint is a display marker only; before/after means refer to the local parent segment at selection.
    split, reduction, start, end = candidate
    left_time = observed['time_center'].iloc[split - 1]
    right_time = observed['time_center'].iloc[split]
    return dict(rank=rank, left_time=left_time, right_time=right_time,
                boundary_time=pd.Timestamp(int(left_time.value) + (int(right_time.value) - int(left_time.value)) // 2, tz='UTC'),
                left_n=split - start, right_n=end - split,
                mean_before=_stable_mean(observed['signal'].iloc[start:split].to_numpy(dtype=float)),
                mean_after=_stable_mean(observed['signal'].iloc[split:end].to_numpy(dtype=float)),
                reduction_normalized=reduction, penalty_normalized=penalty)


def _changes(series, config):
    # CORE LOGIC: STEP 1 — Establish an exploratory global penalty on the centered normalized signal.
    # Input: signal=[0,0,4,4], change_penalty=3.
    # Output: normalized=[-1,-1,1,1], variance=1, penalty_normalized=3*log(4)≈4.158883.
    # Explanation: A split must reduce normalized SSE by more than this fixed variance-and-size heuristic.
    # Trick: The penalty is dimensionless and is neither a p-value nor a calibrated false-alarm threshold.
    observed = series.loc[series['supported']].reset_index(drop=True)
    normalized = _centered_scale(observed['signal'].to_numpy(dtype=float))
    penalty = float(config.change_penalty * np.var(normalized) * np.log(len(normalized))) if len(normalized) else np.nan
    context = dict(penalty_normalized=penalty, changes_found=0)
    empty = pd.DataFrame(columns=CHANGE_COLUMNS)
    # VALIDATION LOGIC: Explain disabled, sparse and constant populations before scanning candidate boundaries.
    if config.max_changes == 0:
        return empty, dict(context, change_status='disabled')
    if len(observed) < 2 * config.min_segment_bins:
        return empty, dict(context, change_status='insufficient_bins')
    if not np.any(normalized):
        return empty, dict(context, change_status='constant_signal')
    # CORE LOGIC: STEP 2 — Greedily choose the largest eligible mean-shift improvement from the current segments.
    # Input: normalized=[-1,-1,1,1], min_segment_bins=2, penalty=1, max_changes=1.
    # Output: prefix=[0,-1,-2,-1,0], best=(split=2,reduction=4,start=0,end=4), which exceeds penalty=1.
    # Explanation: The current segment with the largest SSE reduction above the penalty is selected for splitting.
    # Trick: Prefix sums make each pass linear; the cap bounds work, and tied gains choose the earliest segment/split.
    prefix = np.concatenate(([0.0], np.cumsum(normalized)))
    segments, records = [(0, len(observed))], []
    for rank in range(1, config.max_changes + 1):
        candidates = [_best_split(prefix, start, end, config.min_segment_bins) for start, end in segments]
        candidates = [candidate for candidate in candidates if candidate is not None]
        if not candidates:
            break
        best = max(candidates, key=lambda candidate: candidate[1])
        if best[1] <= penalty:
            break
        # CORE LOGIC: STEP 3 — Record the observed uncertainty bracket and partition only the selected parent.
        # Input: segments=[(0,8)], selected=(split=4,reduction=8,start=0,end=8), rank=1.
        # Output: segments=[(0,4),(4,8)], with one record for the times on either side of index 4.
        # Explanation: Later candidates can split either child while already-selected boundaries stay fixed.
        # Trick: Sorting segment boundaries gives deterministic chronological tie-breaking; records retain selection rank.
        records.append(_change_record(observed, best, rank, penalty))
        split, _, start, end = best
        segments.remove((start, end))
        segments.extend([(start, split), (split, end)])
        segments.sort()
    # RESULT FORMATTING LOGIC: Keep chronological rows and disclose whether any heuristic threshold was crossed.
    result = pd.DataFrame(records, columns=CHANGE_COLUMNS).sort_values('boundary_time').reset_index(drop=True)
    context.update(changes_found=len(result), change_status='candidates_found' if len(result) else 'no_changes_above_penalty')
    return result, context


def _summary(rows, times, series, config, spectrum_context, change_context):
    # CORE LOGIC: STEP 1 — Report temporal eligibility without silently dropping unknown-time evaluation records.
    # Input: total_rows=4, timed_rows=3, n=[2,0,1], supported=[True,False,False], min_bin_count=2.
    # Output: missing_time_rows=1, total_bins=3, occupied_bins=2, supported_bins=1, unsupported_bins=1, missing_bins=1.
    # Explanation: One occupied bin has too little support; a separate empty bin has no timestamped records.
    # Trick: Missing-time rows stay in coverage but cannot enter elapsed-time diagnostics; low-support bins keep raw means.
    occupied = int(series['n'].gt(0).sum())
    supported = int(series['supported'].sum())
    summary = dict(total_rows=len(rows), timed_rows=int(times.notna().sum()), missing_time_rows=int(times.isna().sum()),
                   total_bins=len(series), occupied_bins=occupied, supported_bins=supported,
                   unsupported_bins=occupied - supported, missing_bins=len(series) - occupied)
    # CONFIGURATION LOGIC: Persist the clock, support and interpretation choices alongside the computed diagnostics.
    summary.update(status='ok' if supported else ('no_supported_bins' if len(series) else 'time_unavailable'),
                   frequency=config.frequency, clock='UTC elapsed time', signal=config.signal,
                   bin_seconds=_bin_nanoseconds(config.frequency) / 1e9, min_bin_count=config.min_bin_count,
                   rolling_bins=config.rolling_bins, max_lag=config.max_lag, acf_min_pairs=ACF_MIN_PAIRS,
                   min_segment_bins=config.min_segment_bins, change_penalty=config.change_penalty,
                   max_changes=config.max_changes, change_method='greedy binary segmentation; mean-shift SSE',
                   spectrum_method='floating-mean Lomb-Scargle; equal supported-bin weights',
                   change_time_bounds='adjacent supported bin centers; midpoint is display only')
    summary.update(spectrum_context, **change_context)
    return pd.DataFrame([summary])


def temporal_tables(rows, config=None):
    """Return temporal tables for prepared candidate rows, including unknown-time coverage.

    ``signal`` selects signed residual bias or mean absolute error. Bin starts are
    left-inclusive and anchored to the Unix epoch in UTC. Gaps remain missing.
    The scan needs 20 supported bins and caps its longest period at half the span.
    Mean-shift candidates use observed-bin counts, an explicit heuristic penalty
    and a hard candidate cap; they do not trigger retraining or model selection.
    """
    # VALIDATION LOGIC: Consume prepared finite errors only; this layer never manufactures residuals or timestamps.
    config = TemporalConfig() if config is None else config
    if not isinstance(config, TemporalConfig):
        raise TypeError('config must be a TemporalConfig instance or None.')
    required = {'__error_candidate', '__ae_candidate', '__entity'}
    if not required.issubset(rows):
        raise ValueError(f'Prepared temporal rows need {sorted(required)}.')
    if not np.isfinite(rows[['__error_candidate', '__ae_candidate']].to_numpy(dtype=float)).all():
        raise ValueError('Temporal diagnostics require finite prepared candidate errors.')
    # CORE LOGIC: STEP 1 — Compute each diagnostic from the same full UTC grid and selected supported signal.
    # Input: prepared errors=[-1,3,2], time=[Jan 1 01:00,Jan 1 23:00,NaT] UTC, frequency='1D'.
    # Output: series has n=[2], bias=[1], mae=[2]; summary has total_rows=3, timed_rows=2, missing_time_rows=1;
    #         autocorrelation/spectrum/change_points are empty, with insufficient-bin statuses on the latter two.
    # Explanation: Temporal analyses use the two known-time records while preserving the untimed record in coverage.
    # Trick: All diagnostics share support masking and elapsed-time alignment; none fills missing bins with zeros.
    times = _time_values(rows)
    series = _series(rows, times, config) if times.notna().any() else pd.DataFrame(columns=SERIES_COLUMNS)
    spectrum, spectrum_context = _spectrum(series, config)
    changes, change_context = _changes(series, config)
    summary = _summary(rows, times, series, config, spectrum_context, change_context)
    return dict(summary=summary, series=series, autocorrelation=_autocorrelation(series, config),
                spectrum=spectrum, change_points=changes)
