"""Readable, explicitly exploratory plots for irregular-trade temporal diagnostics."""
# SETUP LOGIC: Unmanaged figures inherit shared local styling without changing notebook rcParams.
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator
from .plot_style import (make_figure, finish_figure, sequence_ticks, figure_title,
                         TEAL, SLATE, ORANGE)

# CONFIGURATION LOGIC: Status wording states the observation limit instead of inventing a conclusion.
STATUS_MESSAGES = {
    'time_unavailable': 'No usable timestamps',
    'no_supported_bins': 'No bins meet minimum support',
    'insufficient_bins': 'Too few supported bins',
    'constant_signal': 'The supported signal is constant',
    'inadequate_period_range': 'No period range spans at least two cycles',
    'numerically_unavailable': 'Spectrum could not be resolved numerically',
    'disabled': 'Mean-shift scan disabled',
    'no_changes_above_penalty': 'No mean-shift candidates exceed the chosen penalty',
    'no_identified_timed_events': 'No events have both an entity and a timestamp',
    'no_within_entity_pairs': 'No within-entity event pairs are available',
}


def _empty(axis, title, message):
    # PLOTTING LOGIC: Unavailable diagnostics carry an explicit explanation instead of a zero curve.
    axis.set_title(title)
    axis.text(.5, .5, message, transform=axis.transAxes, ha='center', va='center',
              color=SLATE, fontsize=10, wrap=True)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_axis_off()


def _value_scale(values):
    # PLOTTING LOGIC: Rescale extreme residual axes only for display; tables retain the declared units.
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    maximum = float(np.max(np.abs(finite))) if len(finite) else 0.0
    return (float(10.0 ** np.floor(np.log10(maximum))) or maximum) if maximum and (maximum > 1e5 or maximum < 1e-3) else 1.0


def _time_ticks(axis, series, frequency_seconds):
    # PLOTTING LOGIC: Positions index the complete fixed-duration grid, so absent dates keep their elapsed width.
    stamp_format = '%Y-%m-%d\n%H:%M' if frequency_seconds < 86400 else '%Y-%m-%d'
    if frequency_seconds < 60:
        stamp_format = '%Y-%m-%d\n%H:%M:%S'
    sequence_ticks(axis, series['time'].dt.strftime(stamp_format).tolist(), max_ticks=5)
    axis.set_xlabel('Bin start · UTC')


def _series_panel(axis, series, changes, summary, unit):
    # PLOTTING LOGIC: Separate raw bin signal from its trailing average; missing raw bins break the line.
    if series.empty or not summary['supported_bins']:
        _empty(axis, 'Residuals through time', STATUS_MESSAGES.get(summary['status'], summary['status']))
        return
    values = series['signal'].to_numpy(dtype=float)
    scale = _value_scale(values)
    x = np.arange(len(series))
    signal_name = 'Signed bias' if summary['signal'] == 'bias' else 'Mean absolute error'
    axis.plot(x, values / scale, color=TEAL, linewidth=.8, alpha=.5, marker='.', markersize=2,
              label=f'{signal_name} per supported bin')
    axis.plot(x, series['rolling_signal'].to_numpy(dtype=float) / scale, color=TEAL, linewidth=1.8,
              label=f"Trailing {int(summary['rolling_bins'])} bins · available-bin mean")
    if summary['signal'] == 'bias':
        axis.axhline(0, color=SLATE, linewidth=.7)
    # PLOTTING LOGIC: Shade the observed bracket; a midpoint marker never implies an exactly located shift.
    for index, change in changes.iterrows():
        first = series['time_center'].iloc[0]
        left = (int(change['left_time'].value) - int(first.value)) / 1e9 / summary['bin_seconds']
        right = (int(change['right_time'].value) - int(first.value)) / 1e9 / summary['bin_seconds']
        axis.axvspan(left, right, color=ORANGE, alpha=.14)
        axis.axvline((left + right) / 2, color=ORANGE, linestyle='--', linewidth=1,
                     label='Mean-shift candidate · observed bracket' if index == 0 else None)
    label = unit if scale == 1 else f'{unit} / {scale:.0e}'
    axis.set(title=f'{signal_name} through time | {int(summary["changes_found"])} shift candidates', ylabel=label)
    _time_ticks(axis, series, summary['bin_seconds'])
    axis.legend(loc='best', fontsize=8)


def _coverage_panel(axis, series, summary):
    # PLOTTING LOGIC: Occupancy makes calendar gaps and changing evaluation volume visible beside the residuals.
    if series.empty:
        _empty(axis, 'Records per elapsed-time bin', 'No usable timestamps')
        return
    x = np.arange(len(series))
    axis.fill_between(x, series['n'].to_numpy(dtype=float), color=SLATE, alpha=.18, step='mid')
    axis.plot(x, series['n'], color=SLATE, linewidth=1)
    axis.axhline(summary['min_bin_count'], color=ORANGE, linestyle='--', linewidth=1,
                 label=f"Minimum support: {int(summary['min_bin_count']):,} records")
    unsupported = series['n'].gt(0) & ~series['supported']
    if unsupported.any():
        axis.scatter(x[unsupported], series.loc[unsupported, 'n'], color=ORANGE, s=12,
                     label='Occupied bin below minimum', zorder=3)
    axis.set(title=f"Coverage | {int(summary['missing_bins']):,} empty bins · "
                   f"{int(summary['missing_time_rows']):,} untimed records", ylabel='Evaluated records')
    axis.set_ylim(bottom=0)
    _time_ticks(axis, series, summary['bin_seconds'])
    axis.legend(loc='best', fontsize=8)


def _duration_scale(seconds):
    # PLOTTING LOGIC: Human-readable elapsed units are chosen once per panel and stated on its axis.
    if seconds >= 86400:
        return 86400.0, 'days'
    if seconds >= 3600:
        return 3600.0, 'hours'
    if seconds >= 60:
        return 60.0, 'minutes'
    return 1.0, 'seconds'


def _lag_ticks(axis, lags):
    # PLOTTING LOGIC: Event and bin lags are discrete counts; show actual integer positions on both short and long scans.
    values = np.asarray(lags, dtype=int)
    indices = np.unique(np.linspace(0, len(values) - 1, min(8, len(values)), dtype=int))
    ticks = values[indices]
    axis.set_xticks(ticks, [str(value) for value in ticks])
    axis._comparison_category_x = True


def _spectrum_panel(axis, spectrum, summary):
    # PLOTTING LOGIC: Overlay the cadence window as an alias clue; neither line carries significance thresholds.
    title = 'Lomb–Scargle spectrum | descriptive'
    if spectrum.empty:
        status = summary['spectrum_status']
        message = STATUS_MESSAGES.get(status, status)
        if status == 'insufficient_bins':
            message += f"\n{int(summary['supported_bins'])} supported; need {int(summary['spectrum_min_bins'])}"
        _empty(axis, title, message)
        return
    ordered = spectrum.sort_values('period_seconds')
    scale, duration_unit = _duration_scale(float(ordered['period_seconds'].max()))
    x = ordered['period_seconds'].to_numpy(dtype=float) / scale
    axis.plot(x, ordered['power'], color=TEAL, linewidth=1.6, label='Residual power · floating mean')
    axis.plot(x, ordered['sampling_window'], color=SLATE, linewidth=1, linestyle='--',
              label='Sampling window · cadence aliases')
    peak_period = summary['spectrum_peak_period_seconds'] / scale
    axis.scatter([peak_period], [summary['spectrum_peak_power']], color=ORANGE, s=24, zorder=3)
    axis.set(title=f'{title}\nLargest scanned peak: {peak_period:.3g} {duration_unit}',
             xlabel=f'Period · elapsed {duration_unit}', ylabel='Normalized power')
    axis.set_xscale('log')
    # PLOTTING LOGIC: Show readable period values across short ranges instead of a lone power-of-ten label.
    axis.xaxis.set_major_locator(LogLocator(base=2, numticks=8))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_ylim(bottom=0)
    axis.legend(loc='best', fontsize=8)


def _acf_panel(axis, autocorrelation, summary):
    # PLOTTING LOGIC: Fixed elapsed-time lags preserve missing-bin separation; no iid confidence band is implied.
    if autocorrelation.empty:
        _empty(axis, 'Correlation at fixed elapsed-time lags', 'No eligible elapsed-time lags')
        return
    table = autocorrelation
    axis.bar(table['lag_bins'], table['correlation'], color=TEAL, alpha=.8, width=.7)
    axis.axhline(0, color=SLATE, linewidth=.7)
    scale, duration_unit = _duration_scale(summary['bin_seconds'])
    bin_duration = summary['bin_seconds'] / scale
    minimum, maximum = int(table['pair_n'].min()), int(table['pair_n'].max())
    axis.set(title=f'Gap-aware lag correlation | {minimum:,}–{maximum:,} pairs per lag',
             xlabel=f'Lag in bins · one bin = {bin_duration:g} {duration_unit}', ylabel='Pearson correlation')
    axis.set_ylim(-1.05, 1.05)
    _lag_ticks(axis, table['lag_bins'])
    if not table['correlation'].notna().any():
        axis.text(.5, .6, f"Need ≥{int(summary['acf_min_pairs'])} pairs and variation on both sides",
                  transform=axis.transAxes, ha='center', color=SLATE, fontsize=9)


def temporal_figure(tables, *, name='Candidate', unit='units'):
    """Return a 2×2 figure of residual timing, occupancy, spectrum and fixed-clock correlation."""
    # PLOTTING LOGIC: Tables carry all calculations; the figure adds labels and presentation without new inference.
    figure = make_figure((14, 10))
    axes = figure.subplots(2, 2)
    summary = tables['summary'].iloc[0]
    _series_panel(axes[0, 0], tables['series'], tables['change_points'], summary, unit)
    _coverage_panel(axes[0, 1], tables['series'], summary)
    _spectrum_panel(axes[1, 0], tables['spectrum'], summary)
    _acf_panel(axes[1, 1], tables['autocorrelation'], summary)
    subtitle = (f"{summary['frequency']} UTC elapsed bins · {int(summary['supported_bins']):,} supported / "
                f"{int(summary['total_bins']):,} total · equal supported-bin weights\n"
                'Exploratory peaks and mean-shift candidates; no significance or retraining decision')
    figure_title(figure, f'{name} · temporal residual diagnostics', subtitle)
    return finish_figure(figure)


def event_lag_figure(table, *, name='Candidate', unit='units'):
    """Display pooled within-entity event lags separately from fixed elapsed-time bin lags."""
    # PLOTTING LOGIC: Successive events are an observation clock; disclose the elapsed gaps those pairs span.
    figure = make_figure((13, 5))
    axes = figure.subplots(1, 2)
    available = table.loc[table['pair_n'].gt(0)]
    if available.empty:
        status = table['status'].iloc[0] if len(table) else 'no_within_entity_pairs'
        _empty(axes[0], 'Within-entity event-lag correlation', STATUS_MESSAGES.get(status, status))
        _empty(axes[1], 'Elapsed gaps behind event lags', 'No within-entity event pairs are available')
    else:
        _event_panels(axes, available)
    figure_title(figure, f'{name} · within-entity event timing',
                 'Equal-time events averaged per entity · pooled correlation can include entity biases\n'
                 'Event spacing varies; inspect entity filters and the elapsed gaps alongside correlation')
    return finish_figure(figure)


def _event_panels(axes, table):
    # PLOTTING LOGIC: Pair counts and contributing entities expose the population behind each event lag.
    minimum, maximum = int(table['pair_n'].min()), int(table['pair_n'].max())
    entity_min, entity_max = int(table['entity_n'].min()), int(table['entity_n'].max())
    axes[0].bar(table['lag_events'], table['correlation'], width=.7, color=TEAL, alpha=.8)
    axes[0].axhline(0, color=SLATE, linewidth=.7)
    axes[0].set(title=f'Within-entity pairs | {minimum:,}–{maximum:,} per lag\n'
                      f'{entity_min:,}–{entity_max:,} contributing entities',
                xlabel='Lag in distinct within-entity events', ylabel='Pooled Pearson correlation', ylim=(-1.05, 1.05))
    _lag_ticks(axes[0], table['lag_events'])
    if not table['correlation'].notna().any():
        axes[0].text(.5, .6, 'Correlation unavailable for the observed pair values',
                     transform=axes[0].transAxes, ha='center', color=SLATE, fontsize=9)
    # PLOTTING LOGIC: Median and P90 gaps are descriptive durations, never confidence limits on correlation.
    scale, duration_unit = _duration_scale(float(table['p90_gap_seconds'].max()))
    axes[1].plot(table['lag_events'], table['median_gap_seconds'] / scale, color=TEAL, linewidth=1.7,
                 marker='o', markersize=3, label='Median elapsed gap')
    axes[1].plot(table['lag_events'], table['p90_gap_seconds'] / scale, color=SLATE, linewidth=1.2,
                 linestyle='--', label='P90 elapsed gap')
    axes[1].set(title='Elapsed time represented by each event lag',
                xlabel='Lag in distinct within-entity events', ylabel=f'Elapsed {duration_unit}')
    axes[1].set_ylim(bottom=0)
    _lag_ticks(axes[1], table['lag_events'])
    axes[1].legend(loc='best', fontsize=8)
