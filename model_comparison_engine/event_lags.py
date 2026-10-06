"""Descriptive within-entity event-lag correlations, retaining elapsed-time context."""
# SETUP LOGIC: Event-lag pairs are distinct from fixed-clock residual diagnostics.
import numpy as np
import pandas as pd

# CONFIGURATION LOGIC: Keep unsupported outputs machine-readable and stable.
COLUMNS = ['lag_events','pair_n','entity_n','correlation','median_gap_seconds',
           'p90_gap_seconds','status','source_rows','eligible_rows','distinct_events','tied_rows_collapsed']


def _correlation(left, right):
    # CORE LOGIC: STEP 1 — Scale each paired vector before centering to keep covariance finite.
    # Input: left=[1e200,2e200,3e200], right=[2e200,4e200,6e200].
    # Output: centered left=right≈[-1/3,0,1/3], correlation≈1.
    # Explanation: Scaling does not change Pearson correlation, while direct squared sums could overflow.
    # Trick: Fewer than three pairs or a constant side is unassessed, not a measured zero correlation.
    if len(left) < 3:
        return np.nan
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    a = a/max(float(np.max(np.abs(a))), np.finfo(float).tiny)
    b = b/max(float(np.max(np.abs(b))), np.finfo(float).tiny)
    a, b = a-a.mean(), b-b.mean()
    denominator = np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.clip(np.dot(a,b)/denominator,-1,1)) if denominator > 0 else np.nan


def _events(rows, signal):
    # CORE LOGIC: STEP 1 — Select timed, identified events and a consistently defined residual signal.
    # Input: entities=['A','A',None], times=['09:00','09:00','09:01'], errors=[1,3,7], signal='bias'.
    # Output: eligible entities=['A','A'], values=[1,3], source_rows=3, eligible_rows=2.
    # Explanation: An unknown entity cannot safely be linked to a prior event from the same entity.
    # Trick: Removing missing identifiers is specific to this temporal diagnostic, not to model scoring.
    valid = rows['__time'].notna() & rows['__entity'].notna()
    field = '__error_candidate' if signal == 'bias' else '__ae_candidate'
    selected = rows.loc[valid, ['__entity','__time',field]].rename(columns={field:'value'})
    # CORE LOGIC: STEP 2 — Collapse simultaneous predictions before defining an ordered event lag.
    # Input: (entity,time,value)=[('A','09:00',1),('A','09:00',3),('A','09:01',4)].
    # Output: [('A','09:00',2),('A','09:01',4)]; tied_rows_collapsed=1.
    # Explanation: The two 09:00 predictions define one mean event, so file order cannot invent lead/lag.
    # Trick: Each simultaneous group has its own scale, so an unrelated huge residual cannot erase tiny events.
    selected['scale'] = selected['value'].abs()
    selected['scale'] = selected.groupby(['__entity','__time'],sort=False,observed=True)['scale'].transform('max')
    selected['value'] = selected['value']/selected['scale'].where(selected['scale'].gt(0),1.0)
    events = selected.groupby(['__entity','__time'],sort=False,observed=True).agg(value=('value','mean'),scale=('scale','first')).reset_index()
    events['value'] *= events.pop('scale')
    events = events.sort_values('__time', kind='stable').reset_index(drop=True)
    return events, dict(source_rows=len(rows), eligible_rows=len(selected), distinct_events=len(events),
                       tied_rows_collapsed=len(selected)-len(events))


def _elapsed(previous, current):
    # CORE LOGIC: STEP 1 — Measure elapsed seconds using unbounded timestamp subtraction.
    # Input: previous=['2026-01-01 09:00:00+00:00'], current=['2026-01-01 09:02:00+00:00'], index=[7].
    # Output: a Series with index=[7] and seconds=[120.0].
    # Explanation: Nanosecond timestamps are subtracted before conversion to seconds.
    # Trick: Python integers avoid Timedelta's roughly 292-year overflow; retained indices preserve pair alignment.
    return pd.Series([(int(right.value)-int(left.value))/1e9 for left,right in zip(previous,current)],
                     index=current.index,dtype=float)


def event_lag_table(rows, max_lag=20, signal='bias'):
    """Pool same-entity lagged pairs; includes entity-level bias and unequal event counts.

    These are descriptive correlations, not an IID whiteness test. Use a fixed
    entity/population filter to distinguish serial persistence from composition.
    """
    # VALIDATION LOGIC: Bound work and reject ambiguous signal choices before building shifted pairs.
    if isinstance(max_lag,bool) or not isinstance(max_lag,(int,np.integer)) or not 0 <= max_lag <= 1000:
        raise ValueError('max_lag must be an integer from 0 through 1000.')
    if signal not in {'bias','mae'}:
        raise ValueError('signal must be bias or mae.')
    if max_lag == 0:
        return pd.DataFrame(columns=COLUMNS)
    events, audit = _events(rows, signal)
    # REPORTING LOGIC: Unknown identity/time is an unsupported diagnostic, not a zero-valued correlation.
    if events.empty:
        result = dict(lag_events=1,pair_n=0,entity_n=0,correlation=np.nan,
                      median_gap_seconds=np.nan,p90_gap_seconds=np.nan,status='no_identified_timed_events',**audit)
        return pd.DataFrame([result], columns=COLUMNS)
    records = []
    # CORE LOGIC: STEP 1 — Shift each entity separately and measure the actual time spanned by each lag.
    # Input: events=[('A','09:00',1),('B','09:01',9),('A','09:02',3)], lag_events=1.
    # Output: one eligible pair A:(1,3), pair_n=1, entity_n=1, median_gap_seconds=120.
    # Explanation: B's event is not paired with A; the interval can differ across entities and events.
    # Trick: Grouped shift preserves identity, while the finite mask aligns values and elapsed times by row index.
    for lag in range(1, min(max_lag, int(events.groupby('__entity', observed=True).size().max())-1)+1):
        previous = events.groupby('__entity', sort=False, observed=True)[['value','__time']].shift(lag)
        valid = previous['value'].notna()
        elapsed = _elapsed(previous.loc[valid,'__time'],events.loc[valid,'__time'])
        correlation = _correlation(previous.loc[valid,'value'], events.loc[valid,'value'])
        records.append(dict(lag_events=lag,pair_n=int(valid.sum()),entity_n=events.loc[valid,'__entity'].nunique(),
                            correlation=correlation,median_gap_seconds=elapsed.median(),p90_gap_seconds=elapsed.quantile(.9),
                            status='ok' if np.isfinite(correlation) else 'insufficient_pairs_or_constant',**audit))
    # REPORTING LOGIC: A singleton per entity has no lagged pairs; retain its coverage audit explicitly.
    if not records:
        records.append(dict(lag_events=1,pair_n=0,entity_n=0,correlation=np.nan,
                            median_gap_seconds=np.nan,p90_gap_seconds=np.nan,status='no_within_entity_pairs',**audit))
    return pd.DataFrame(records, columns=COLUMNS)
