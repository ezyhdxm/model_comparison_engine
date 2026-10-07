"""Evidence-linked reading guidance for descriptive temporal diagnostics."""
# SETUP LOGIC: Interpret existing tables without running extra tests or ranking models.
import numpy as np
import pandas as pd


def _duration(seconds):
    # PRESENTATION LOGIC: Preserve approximate scale while avoiding false timestamp precision.
    for divisor, label in ((86400, 'days'), (3600, 'hours'), (60, 'minutes')):
        if seconds >= divisor:
            return f'{seconds / divisor:.3g} {label}'
    return f'{seconds:.3g} seconds'


def _lag_guidance(table):
    # CORE LOGIC: STEP 1 — Select the largest absolute finite descriptive correlation with its actual support.
    # Input: lag_bins=[1,2,3], correlation=[0.1,NaN,-0.6], pair_n=[20,0,12], lag_seconds=[60,120,180].
    # Output: selected lag=3, correlation=-0.6, pair_n=12, elapsed lag=180 seconds.
    # Explanation: Missing or constant-pair results cannot win the descriptive ranking.
    # Trick: Absolute size is used only to locate a reading example; searching many lags does not supply significance.
    known = table.loc[np.isfinite(pd.to_numeric(table['correlation'], errors='coerce'))]
    selected = known.loc[known['correlation'].abs().idxmax()] if len(known) else None
    # INTERPRETATION LOGIC: The guidance explicitly distinguishes persistence from alternating residuals.
    if selected is None:
        return ('No lag has enough varying paired observations for correlation.',
                'Unavailable correlation is not evidence that residuals are independent.',
                'Use a longer evaluation interval or a coarser bin; inspect support counts first.')
    sign = 'same-direction persistence' if selected['correlation'] > 0 else 'alternation between above/below-average bin values'
    return (f"Largest absolute scanned correlation: {selected['correlation']:.3g} at "
            f"{_duration(selected['lag_seconds'])}, using {int(selected['pair_n']):,} bin pairs.",
            f'The sign describes {sign}; this is an exploratory maximum, without a significance claim.',
            'Check whether it persists in another interval and within individual entities; changing entity mix can drive it.')


def _spectrum_guidance(summary, spectrum):
    # INTERPRETATION LOGIC: A status is an availability condition, never a verdict about predictability.
    if spectrum.empty:
        return (f"Spectrum unavailable: {summary['spectrum_status']}; {int(summary['supported_bins'])} supported bins.",
                'The scan needs enough variable supported bins and at least two cycles in its scanned period range.',
                'Inspect occupancy; use more history before interpreting cycles. Do not fill missing bins with zero.')
    # CORE LOGIC: STEP 1 — Compare the reported residual peak with the sampling window at the same period.
    # Input: period_seconds=[86400,172800], power=[0.2,0.6], sampling_window=[0.3,0.1], peak=172800.
    # Output: residual peak period=172800 seconds, residual power=0.6, sampling-window power there=0.1.
    # Explanation: The matched window value checks sampling cadence at precisely the highlighted residual period.
    # Trick: These normalized quantities are descriptive scores, not p-values or percent prediction improvement.
    peak = float(summary['spectrum_peak_period_seconds'])
    match = spectrum.loc[(spectrum['period_seconds'] - peak).abs().idxmin()]
    # INTERPRETATION LOGIC: Cadence confounding remains possible even when the exact peak's window power is small.
    return (f"Largest scanned peak: {_duration(peak)}, normalized residual power {match['power']:.3g}; "
            f"sampling-window power at that period {match['sampling_window']:.3g}.",
            'A sinusoid at this period fits the selected bin signal relatively well. It does not prove a cycle or forecasting gain.',
            'Compare the full sampling-window curve and weekday/hour coverage; weekly trading calendars can create aliases. '
            'Check another holdout interval before adding a periodic feature.')


def temporal_interpretation(tables, *, unit='units'):
    """Return compact observations, meanings and next checks grounded in the supplied tables."""
    # VALIDATION LOGIC: Require computed temporal output rather than accepting arbitrary UI labels as evidence.
    summary_table = tables.get('summary', pd.DataFrame())
    if summary_table.empty:
        return pd.DataFrame(columns=['topic', 'observation', 'interpretation', 'next_check'])
    summary = summary_table.iloc[0]
    # INTERPRETATION LOGIC: Keep candidate-only and paired population identities explicit in exported prose.
    records = [dict(topic='Population and clock',
                    observation=f"{int(summary['timed_rows']):,} timed records; {int(summary['missing_time_rows']):,} untimed. "
                                f"{int(summary['supported_bins']):,} supported / {int(summary['total_bins']):,} elapsed bins; "
                                f"{int(summary['missing_bins']):,} empty; frequency {summary['frequency']}.",
                    interpretation='These are fixed UTC elapsed bins. Empty bins remain missing and bin statistics receive equal weight in timing diagnostics.',
                    next_check='Check population, entity mix and occupancy before comparing patterns. Untimed rows still count in prediction coverage.')]
    signal = 'signed prediction minus actual' if summary['signal'] == 'bias' else 'mean absolute prediction error'
    records.append(dict(topic='Residual signal', observation=f"Displayed signal is {summary['signal']} ({unit}): {signal}.",
                        interpretation='Positive bias means overprediction; negative means underprediction. Small bias can hide large offsetting errors. '
                                       'MAE measures error magnitude and cannot be negative.',
                        next_check='Compare bias with MAE, quantity and entity slices. The rolling curve averages available supported bins, not all records.'))
    # DIAGNOSTIC LOGIC: Read numerical evidence from the existing lag and spectrum outputs.
    lag = _lag_guidance(tables['autocorrelation'])
    spectrum = _spectrum_guidance(summary, tables['spectrum'])
    records.extend(dict(topic=topic, observation=values[0], interpretation=values[1], next_check=values[2])
                   for topic, values in [('Fixed-clock correlation', lag), ('Lomb–Scargle spectrum', spectrum)])
    # INTERPRETATION LOGIC: Change candidates are offline, heuristic and not causal break dates.
    records.append(dict(topic='Mean-shift candidates',
                        observation=f"{int(summary['changes_found'])} candidates; status {summary['change_status']}.",
                        interpretation='The scan searches for shifts in the selected binned mean using a heuristic penalty. '
                                       'Zero candidates does not prove stability; a candidate does not identify its cause.',
                        next_check='Inspect the before/after windows, entity/quantity mix and model/data changes. '
                                   'The shaded bracket locates adjacent supported bins, not an exact event timestamp.'))
    events = tables.get('event_autocorrelation', pd.DataFrame())
    if len(events) and events['pair_n'].gt(0).any():
        records.append(_event_guidance(events))
    return pd.DataFrame(records)


def _event_guidance(events):
    # CORE LOGIC: STEP 1 — Read the smallest available within-entity event lag and its elapsed-time support.
    # Input: lag_events=[1,2], pair_n=[100,80], median_gap_seconds=[60,150], entity_n=[5,5].
    # Output: selected lag_events=1, pair_n=100, median elapsed gap=60 seconds, contributing entities=5.
    # Explanation: Event lag 1 means the next distinct timestamp in each entity's selected evaluation records.
    # Trick: It does not necessarily mean the next market trade; unknown predictions or active filters change the event population.
    selected = events.loc[events['pair_n'].gt(0)].sort_values('lag_events').iloc[0]
    # INTERPRETATION LOGIC: Distinguish observation order from a fixed elapsed clock and from demeaned residuals.
    return dict(topic='Within-entity events',
                observation=f"Event lag {int(selected['lag_events'])}: {int(selected['pair_n']):,} pairs across "
                            f"{int(selected['entity_n']):,} entities; median elapsed gap {_duration(selected['median_gap_seconds'])}.",
                interpretation='Pairs stay within the mapped entity and equal-time rows are averaged. '
                               'The pooled correlation is not entity-demeaned and active entities contribute more pairs.',
                next_check='Compare median/P90 elapsed gaps and filter one entity; persistent entity bias can mimic within-entity memory.')
