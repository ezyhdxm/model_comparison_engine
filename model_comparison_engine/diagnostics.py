"""Descriptive diagnostics on every valid candidate record, independent of a reference."""
# SETUP LOGIC: Prepared rows already contain adjusted predictions and errors in the requested unit.
import numpy as np
import pandas as pd
from .slices import Slice, slice_labels

# CONFIGURATION LOGIC: Explicit columns keep empty results usable by reports and notebooks.
ERROR_COLUMNS = ['n', 'candidate_mae', 'candidate_rmse', 'candidate_bias',
                 'candidate_median_ae', 'candidate_p90', 'candidate_p95', 'candidate_p99',
                 'candidate_within_tol_pct', 'candidate_under_pct', 'candidate_over_pct',
                 'candidate_exact_pct', 'low_support']
SUMMARY_COLUMNS = [*ERROR_COLUMNS, 'entities', 'dates', 'missing_date_rows',
                   'tail_threshold', 'tail_n', 'tail_pct', 'tail_error_share_pct']
CALIBRATION_COLUMNS = ['bin', 'n', 'predicted_min', 'predicted_max', 'mean_prediction',
                       'mean_actual', 'candidate_bias', 'candidate_mae']
SLICE_COLUMNS = ['spec', 'column', 'group', *ERROR_COLUMNS, 'error_share_pct',
                 'tail_n', 'tail_error_share_pct']
QUANTILE_LEVELS = [.01, .05, .25, .5, .75, .95, .99]
CASE_COLUMNS = ['__actual', '__candidate', 'signed_error', 'abs_error', '__date', '__entity']


def _stable_mean(values):
    # CORE LOGIC: STEP 1 — Normalize before averaging to avoid overflowing a finite mean.
    # Input: values=[-1e308,1e308].
    # Output: 0.0.
    # Explanation: The normalized values [-1,1] cancel before their mean is converted back to source units.
    # Trick: Maximum magnitude bounds each summand by one; empty inputs have no defined mean.
    array = np.asarray(values, dtype=float)
    scale = np.max(np.abs(array)) if len(array) else np.nan
    return float(np.mean(array / scale) * scale) if scale > 0 else float(scale)


def _quantiles(values, probabilities):
    # CORE LOGIC: STEP 1 — Interpolate bounded values, then restore the original error units.
    # Input: values=[-1e308,1e308], probabilities=[.25,.5,.75].
    # Output: [-5e307,0,5e307] (subject to floating-point rounding).
    # Explanation: The three interpolated normalized quantiles are [-.5,0,.5], each multiplied by 1e308.
    # Trick: Direct subtraction between extreme signed endpoints can overflow during interpolation.
    array = np.asarray(values, dtype=float)
    scale = np.max(np.abs(array)) if len(array) else np.nan
    if not len(array):
        return np.full(len(probabilities), np.nan)
    return np.quantile(array / scale, probabilities) * scale if scale > 0 else np.zeros(len(probabilities))


def _error_stats(rows, tolerance, min_count):
    # CORE LOGIC: STEP 1 — Compute average and tail losses from the candidate's signed errors.
    # Input: errors=[-1,3], tolerance=1, min_count=3.
    # Output: n=2, MAE=2, RMSE≈2.236068, bias=1, median_AE=2, P90=2.8, P95=2.9, P99=2.98.
    # Explanation: Absolute losses [1,3] average to 2; squared losses [1,9] average to 5 before the square root.
    # Trick: Squaring normalized errors avoids overflow; all quantities retain the caller's error unit.
    error = rows['__error_candidate'].to_numpy(dtype=float)
    absolute = rows['__ae_candidate'].to_numpy(dtype=float)
    scale = np.max(absolute) if len(rows) else np.nan
    normalized = error / scale if scale > 0 else error
    rmse = float(np.sqrt(np.mean(normalized ** 2)) * scale) if len(rows) else np.nan
    quantiles = _quantiles(absolute, [.5, .9, .95, .99])
    result = dict(n=len(rows), candidate_mae=_stable_mean(absolute), candidate_rmse=rmse,
                  candidate_bias=_stable_mean(error))
    result.update(zip(['candidate_median_ae', 'candidate_p90', 'candidate_p95', 'candidate_p99'], quantiles))
    # CORE LOGIC: STEP 2 — Count directional errors and tolerance matches without dropping tails.
    # Input: errors=[-1,3], tolerance=1, min_count=3.
    # Output: within_tol_pct=50, under_pct=50, over_pct=50, exact_pct=0, low_support=True.
    # Explanation: One of two records is within tolerance, one is below actual, and one is above actual.
    # Trick: Tolerance is inclusive; zero errors are neither underpredictions nor overpredictions.
    indicators = [absolute <= tolerance, error < 0, error > 0, error == 0]
    names = ['candidate_within_tol_pct', 'candidate_under_pct', 'candidate_over_pct', 'candidate_exact_pct']
    result.update({name: float(mask.mean() * 100) if len(rows) else np.nan for name, mask in zip(names, indicators)})
    result['low_support'] = len(rows) < min_count
    return result


def _error_share(absolute, scale, total_weight):
    # CORE LOGIC: STEP 1 — Measure a subset's share of total absolute error without summing large errors.
    # Input: absolute=[1e308], scale=1e308, total_weight=2.
    # Output: 50.0.
    # Explanation: The subset contributes one normalized error unit out of the population's two.
    # Trick: A population of perfect predictions has zero total error; its contribution is defined as 0%.
    return float(np.sum(np.asarray(absolute, dtype=float) / scale) / total_weight * 100) if scale > 0 else 0.0


def _tail_context(rows):
    # CORE LOGIC: STEP 1 — Fix one population-wide tail threshold and error denominator for all slices.
    # Input: absolute=[0,0,10].
    # Output: threshold=9 (approximately), scale=10, total_weight=1.
    # Explanation: Linear P95 interpolation falls between 0 and 10; normalized errors [0,0,1] sum to one.
    # Trick: Tail membership means strictly above P95; tied threshold errors are never arbitrarily split.
    absolute = rows['__ae_candidate'].to_numpy(dtype=float)
    scale = float(np.max(absolute)) if len(rows) else 0.0
    total_weight = float(np.sum(absolute / scale)) if scale > 0 else 0.0
    threshold = float(_quantiles(absolute, [.95])[0])
    return threshold, scale, total_weight


def _summary(rows, tolerance, min_count, context):
    # CORE LOGIC: STEP 1 — Add metadata coverage and the strict P95 tail contribution to candidate losses.
    # Input: errors=[0,0,10], dates=['2026-01-01',None,None], entities=['A','A','B'].
    # Output: entities=2, dates=1, missing_date_rows=2, tail_n=1, tail_pct≈33.3333, tail_error_share_pct=100.
    # Explanation: The only error above P95 is 10, which supplies all absolute error in this population.
    # Trick: Distinct metadata counts exclude missing values; loss metrics still include all three records.
    threshold, scale, total_weight = context
    absolute = rows['__ae_candidate']
    tail = absolute[absolute > threshold]
    result = _error_stats(rows, tolerance, min_count)
    result.update(entities=rows['__entity'].nunique(), dates=rows['__date'].nunique(),
                  missing_date_rows=int(rows['__date'].isna().sum()), tail_threshold=threshold, tail_n=len(tail))
    result.update(tail_pct=len(tail) / len(rows) * 100 if len(rows) else np.nan,
                  tail_error_share_pct=_error_share(tail, scale, total_weight) if len(rows) else np.nan)
    return pd.DataFrame([result], columns=SUMMARY_COLUMNS)


def _calibration(rows, bins):
    # CORE LOGIC: STEP 1 — Form prediction-quantile bins while keeping identical predictions together.
    # Input: predictions=[1,1,3,3], bins=2.
    # Output: edges=[1,2,3], groups=[0,0,1,1].
    # Explanation: The median boundary 2 divides the two prediction levels without splitting either tie.
    # Trick: Duplicate quantile boundaries collapse; an exact interior boundary belongs to the lower bin.
    predictions = rows['__candidate'].to_numpy(dtype=float)
    edges = np.unique(_quantiles(predictions, np.linspace(0, 1, bins + 1))) if len(rows) else np.array([])
    groups = np.searchsorted(edges[1:-1], predictions, side='left')
    grouped = rows.assign(__calibration_group=groups)
    records = []
    # CORE LOGIC: STEP 2 — Average actuals and predictions over the same records in each observed bin.
    # Input: actuals=[0,2,2,4], predictions=[1,1,3,3], errors=[1,-1,1,-1], groups=[0,0,1,1].
    # Output: bins 1/2 have n=[2,2], mean_prediction=[1,3], mean_actual=[1,3], bias=[0,0], MAE=[1,1].
    # Explanation: Each prediction bin has opposite signed errors that cancel in bias while retaining MAE 1.
    # Trick: Calibration levels use native target units; bias and MAE use the already-scaled error unit.
    for index, (_, group) in enumerate(grouped.groupby('__calibration_group', sort=True), start=1):
        records.append(dict(bin=index, n=len(group), predicted_min=group['__candidate'].min(),
                            predicted_max=group['__candidate'].max(), mean_prediction=_stable_mean(group['__candidate']),
                            mean_actual=_stable_mean(group['__actual']), candidate_bias=_stable_mean(group['__error_candidate']),
                            candidate_mae=_stable_mean(group['__ae_candidate'])))
    return pd.DataFrame(records, columns=CALIBRATION_COLUMNS)


def _slice_names(specs):
    # FORMATTING LOGIC: Repeated columns or display names keep distinct identities for different bin specifications.
    names = [spec.name or spec.column for spec in specs]
    return [f'{name} [{index + 1}]' if names.count(name) > 1 else name for index, name in enumerate(names)]


def _worst_slices(rows, specs, tolerance, min_count, top_n, context):
    # CORE LOGIC: STEP 1 — Measure observed slice groups against a shared total-error denominator.
    # Input: segment=['A','A','B'], errors=[0,0,10], min_count=2, P95≈9.
    # Output: A has n=2, MAE=0, error_share=0%, tail_n=0; B has n=1, MAE=10, error_share=100%, tail_n=1.
    # Explanation: Both A records are exact predictions, so B contributes the entire population loss.
    # Trick: A record can belong to several different slice specifications; shares add to 100% within each specification.
    threshold, scale, total_weight = context
    records = []
    for spec, name in zip(specs, _slice_names(specs)):
        grouped = rows.assign(__diagnostic_group=slice_labels(rows, spec))
        for label, group in grouped.groupby('__diagnostic_group', observed=True, sort=True, dropna=False):
            stats = _error_stats(group, tolerance, min_count)
            absolute = group['__ae_candidate']
            records.append(dict(spec=name, column=spec.column, group=str(label), **stats,
                                error_share_pct=_error_share(absolute, scale, total_weight), tail_n=int((absolute > threshold).sum()),
                                tail_error_share_pct=_error_share(absolute[absolute > threshold], scale, total_weight)))
    # CORE LOGIC: STEP 2 — Rank supported groups by MAE, then retain flagged small groups if space remains.
    # Input: groups=['A','B'], MAE=[0,10], low_support=[False,True], top_n=2.
    # Output: ranked groups=['A','B'], with B still visibly flagged as low support.
    # Explanation: A meets minimum support and ranks before B even though B has the larger mean error.
    # Trick: Stable sorting preserves specification/category order when support and MAE tie; ranking never filters source rows.
    result = pd.DataFrame(records, columns=SLICE_COLUMNS)
    return result.sort_values(['low_support', 'candidate_mae'], ascending=[True, False], kind='stable').head(top_n).reset_index(drop=True)


def _worst_cases(rows, specs, top_n, id_column):
    # CORE LOGIC: STEP 1 — Select the largest candidate errors with only the requested case metadata.
    # Input: ids=['a','b','c'], errors=[-2,2,1], top_n=2.
    # Output: ids=['a','b'], signed_error=[-2,2], abs_error=[2,2].
    # Explanation: Both two-unit losses precede the one-unit loss; the tied records retain input order.
    # Trick: Stable sorting retains source order for equal errors, including duplicate index labels; cases do not remove records.
    identity = [column for column in ['__source_position', '__source_index'] if column in rows]
    columns = ([id_column] if id_column is not None else []) + identity + CASE_COLUMNS + [spec.column for spec in specs]
    columns = list(dict.fromkeys(columns))
    cases = rows.assign(signed_error=rows['__error_candidate'], abs_error=rows['__ae_candidate'])
    return cases.sort_values('abs_error', ascending=False, kind='stable').head(top_n).loc[:, columns].copy()


def _daily(rows, tolerance, min_count):
    # CORE LOGIC: STEP 1 — Compute day-level candidate errors for known dates in chronological order.
    # Input: dates=['2026-01-02',None,'2026-01-01'], errors=[2,9,-1].
    # Output: dates=['2026-01-01','2026-01-02'], n=[1,1], MAE=[1,2], bias=[-1,2].
    # Explanation: The two known dates each contain one error, so their biases equal those signed errors.
    # Trick: Unknown dates cannot be plotted chronologically; summary.missing_date_rows accounts for them explicitly.
    records = []
    for day, group in rows.loc[rows['__date'].notna()].groupby('__date', sort=True, observed=True):
        records.append(dict(__date=day, **_error_stats(group, tolerance, min_count)))
    return pd.DataFrame(records, columns=['__date', *ERROR_COLUMNS])


def candidate_tables(rows, slices=None, *, min_count=30, tolerance=1.0, top_n=20, bins=10, id_column=None):
    """Return six candidate-only tables; input rows must already be finite and candidate-valid.

    Signed errors are prediction minus actual. Tail contributions use errors strictly
    above the population P95 and divide by population total absolute error. Slice
    ranking places supported groups first; the top limit affects display only.
    """
    # VALIDATION LOGIC: Reject ambiguous limits or missing canonical columns before computing diagnostics.
    required = {'__actual', '__candidate', '__error_candidate', '__ae_candidate', '__date', '__entity'}
    if required - set(rows):
        raise ValueError(f'Missing prepared candidate columns: {sorted(required - set(rows))}')
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1 for value in [min_count, top_n, bins]):
        raise ValueError('min_count, top_n and bins must be positive integers.')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative in the displayed error unit.')
    if id_column is not None and id_column not in rows:
        raise ValueError(f'Unknown ID column: {id_column!r}')
    # CONFIGURATION LOGIC: Accept reusable Slice objects or column names without implicit domain assumptions.
    specs = [slices] if isinstance(slices, (Slice, str)) else list(slices or [])
    specs = [Slice(spec) if isinstance(spec, str) else spec for spec in specs]
    if any(not isinstance(spec, Slice) for spec in specs):
        raise TypeError('slices must contain column names or Slice objects.')
    context = _tail_context(rows)
    # REPORTING LOGIC: Each table reads the full candidate population; only ranked display tables are bounded.
    return dict(summary=_summary(rows, tolerance, min_count, context), calibration=_calibration(rows, bins),
                residual_quantiles=pd.DataFrame({'quantile': QUANTILE_LEVELS,
                                                 'signed_error': _quantiles(rows['__error_candidate'], QUANTILE_LEVELS)}),
                worst_slices=_worst_slices(rows, specs, tolerance, min_count, top_n, context),
                worst_cases=_worst_cases(rows, specs, top_n, id_column), daily=_daily(rows, tolerance, min_count))
