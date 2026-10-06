"""Classical candidate residual views with full-population summaries and bounded point displays."""
# SETUP LOGIC: Standard-library normal quantiles avoid requiring a training or statistics dependency.
from statistics import NormalDist
import numpy as np
from .plot_style import (make_figure, finish_figure, figure_title, categorical_ticks,
                         sequence_ticks, row_figure_height)
from .diagnostics import _quantiles

# CONFIGURATION LOGIC: These limits affect drawn points only, never diagnostic table populations.
POINT_LIMIT = 3000
QQ_LIMIT = 500
COLOR = '#187f83'
REFERENCE_COLOR = '#657987'


def _point_indices(n, limit=POINT_LIMIT):
    # CORE LOGIC: STEP 1 — Select evenly spaced source positions for reproducible point displays.
    # Input: n=7, limit=4.
    # Output: [0,2,4,6].
    # Explanation: Four evenly spaced positions include both endpoints of the seven-record input.
    # Trick: Integer positions are independent of duplicate row labels; all positions are kept below the limit.
    return np.linspace(0, n - 1, min(n, limit), dtype=int) if n else np.array([], dtype=int)


def _display_scale(values):
    # PLOTTING LOGIC: Rescale extreme axes before Matplotlib expands limits; table values remain untouched.
    array = np.asarray(values, dtype=float)
    magnitude = float(np.max(np.abs(array))) if len(array) else 0.0
    if magnitude > 0 and (magnitude >= 1e6 or magnitude < 1e-4):
        return float(10.0 ** np.floor(np.log10(magnitude))) or magnitude
    return 1.0


def _unit_label(label, scale):
    # FORMATTING LOGIC: Make any presentation-only axis multiplier explicit beside the unit.
    return label if scale == 1 else f'{label} / {scale:.0e}'


def _empty_axis(axis, title, message):
    # PLOTTING LOGIC: Empty data produce an explicit explanation instead of a misleading zero-valued curve.
    axis.set_title(title)
    axis.text(.5, .5, message, transform=axis.transAxes, ha='center', va='center', color=REFERENCE_COLOR)
    categorical_ticks(axis, [])
    categorical_ticks(axis, [], axis='y')


def _histogram(axis, errors, error_label):
    # PLOTTING LOGIC: The histogram includes every valid error, including records absent from point displays.
    axis.hist(errors, bins=min(50, max(1, int(np.sqrt(len(errors))))), color=COLOR, alpha=.85)
    axis.axvline(0, color=REFERENCE_COLOR, linewidth=1)
    axis.set(title=f'Signed residuals | all {len(errors):,} records', xlabel=error_label, ylabel='Records')


def _calibration(axis, table, scale, target_label):
    # PLOTTING LOGIC: Compare full-population bin means against equality; point area communicates bin support.
    actual = table['mean_actual'].to_numpy(dtype=float) / scale
    predicted = table['mean_prediction'].to_numpy(dtype=float) / scale
    sizes = 25 + 100 * table['n'].to_numpy(dtype=float) / table['n'].max()
    bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    if bounds[0] == bounds[1]:
        padding = max(abs(bounds[0]) * .05, .05)
        bounds = [bounds[0] - padding, bounds[1] + padding]
    axis.plot(bounds, bounds, color=REFERENCE_COLOR, linestyle='--', linewidth=1, label='Prediction = actual')
    axis.plot(predicted, actual, color=COLOR, linewidth=1)
    axis.scatter(predicted, actual, s=sizes, color=COLOR, edgecolors='white', linewidths=.5)
    axis.set(title=f'Calibration | {len(table):,} prediction bins, all records',
             xlabel=f'Mean prediction — {target_label}', ylabel=f'Mean actual — {target_label}')
    axis.legend(fontsize=8)


def _residual_scatter(axis, fitted, errors, target_label, error_label, total):
    # PLOTTING LOGIC: Source-order position sampling is deterministic and explicitly distinguished from population N.
    axis.scatter(fitted, errors, s=9, color=COLOR, alpha=.35, rasterized=True)
    axis.axhline(0, color=REFERENCE_COLOR, linewidth=1)
    axis.set(title=f'Residual vs fitted | {len(errors):,} of {total:,} points',
             xlabel=f'Prediction — {target_label}', ylabel=error_label)


def _qq_coordinates(errors):
    # CORE LOGIC: STEP 1 — Evaluate bounded quantile positions against a descriptive normal reference.
    # Input: errors=[-1,0,1], QQ_LIMIT=500.
    # Output: observed≈[-.666667,0,.666667], normal≈[-.967422,0,.967422].
    # Explanation: Quantile probabilities [1/6,1/2,5/6] locate comparable points in both distributions.
    # Trick: Quantiles use all records; bounding plotted quantiles does not subsample the error population.
    count = min(len(errors), QQ_LIMIT)
    probabilities = (np.arange(count) + .5) / count
    theoretical = np.array([NormalDist().inv_cdf(float(value)) for value in probabilities])
    observed = _quantiles(errors, probabilities)
    return theoretical, observed


def _qq(axis, errors, error_label):
    # PLOTTING LOGIC: The reference line matches median and IQR; it is not a test or a normality requirement.
    theoretical, observed = _qq_coordinates(errors)
    lower, middle, upper = _quantiles(errors, [.25, .5, .75])
    spread = (upper - lower) / (2 * NormalDist().inv_cdf(.75))
    axis.scatter(theoretical, observed, s=10, color=COLOR, alpha=.65, rasterized=True)
    axis.plot(theoretical, middle + spread * theoretical, color=REFERENCE_COLOR, linestyle='--', linewidth=1)
    axis.set(title='Normal Q–Q | descriptive, all records', xlabel='Standard normal quantile', ylabel=error_label)


def _absolute_scatter(axis, fitted, absolute, target_label, error_label, total):
    # PLOTTING LOGIC: Absolute residuals expose error magnitude versus fitted level without assuming constant variance.
    axis.scatter(fitted, absolute, s=9, color=COLOR, alpha=.35, rasterized=True)
    axis.set(title=f'Absolute residual vs fitted | {len(absolute):,} of {total:,} points',
             xlabel=f'Prediction — {target_label}', ylabel=f'Absolute {error_label.lower()}')


def _daily(axis, table, scale, error_label, total):
    # PLOTTING LOGIC: Known-day means retain their original record weights and expose date coverage.
    if table.empty:
        _empty_axis(axis, 'Daily error', 'Date metadata unavailable\nNo records with known dates')
        return
    x = np.arange(len(table))
    marker = '.' if len(table) <= 60 else None
    axis.plot(x, table['candidate_mae'] / scale, color=COLOR, marker=marker, linewidth=1, alpha=.8, label='MAE')
    axis.plot(x, table['candidate_bias'] / scale, color='#c67446', marker=marker, linewidth=1, alpha=.8, label='Bias')
    axis.axhline(0, color=REFERENCE_COLOR, linewidth=.8)
    sequence_ticks(axis, table['__date'], max_ticks=5)
    axis.set(ylabel=error_label, title=f'Daily error | {int(table["n"].sum()):,} of {total:,} dated records')
    axis.legend(fontsize=8)


def candidate_figure(rows, tables, *, name='Candidate', unit='units', error_scale=1.0):
    """Return a 2×3 descriptive figure; the caller supplies tables for these same rows.

    Target levels stay in their source units when error_scale differs from one.
    Only the two scatter displays use bounded deterministic source-position samples.
    """
    # VALIDATION LOGIC: Target-level labels must describe the error conversion used by data preparation.
    if not np.isfinite(error_scale) or error_scale <= 0:
        raise ValueError('error_scale must be a positive finite number.')
    # PLOTTING LOGIC: Unmanaged figures render once in notebooks and save without changing global plot state.
    figure = make_figure((16, 10))
    axes = figure.subplots(2, 3).ravel()
    n = len(rows)
    population = tables['summary']['population'].iloc[0] if 'population' in tables['summary'] else 'candidate'
    population_label = 'paired records' if population == 'paired' else 'candidate-valid records'
    figure_title(figure, f'{name} diagnostics | {n:,} {population_label}',
                 f'Signed error = (prediction − actual) × {error_scale:g}; error unit: {unit}; '
                 f'scatter displays at most {POINT_LIMIT:,} points')
    if not n:
        for axis, title in zip(axes, ['Signed residuals', 'Calibration', 'Residual vs fitted',
                                      'Normal Q–Q', 'Absolute residual vs fitted', 'Daily error']):
            _empty_axis(axis, title, 'No valid candidate records')
        return finish_figure(figure)
    # CORE LOGIC: STEP 1 — Select display positions while leaving full-population aggregates intact.
    # Input: prediction=[1,2,3], error=[0,-1,2], POINT_LIMIT=3000.
    # Output: positions=[0,1,2], selected_predictions=[1,2,3], selected_errors=[0,-1,2].
    # Explanation: Three rows fit below the display limit, so all predictions and residuals appear in each scatter plot.
    # Trick: iloc selects by source position; duplicate source index labels cannot duplicate plotted records.
    positions = _point_indices(n)
    selected = rows.iloc[positions]
    # PLOTTING LOGIC: Scale extreme display ranges before axis calculations while preserving explicit units.
    error_axis_scale = _display_scale(rows['__error_candidate'])
    target_axis_scale = _display_scale(rows[['__candidate', '__actual']].to_numpy().ravel())
    error_label = _unit_label(f'Error ({unit})', error_axis_scale)
    native_label = f'Target level ({unit})' if error_scale == 1 else 'Target level (source units)'
    target_label = _unit_label(native_label, target_axis_scale)
    errors = rows['__error_candidate'].to_numpy(dtype=float) / error_axis_scale
    fitted = selected['__candidate'].to_numpy(dtype=float) / target_axis_scale
    selected_errors = selected['__error_candidate'].to_numpy(dtype=float) / error_axis_scale
    _histogram(axes[0], errors, error_label)
    _calibration(axes[1], tables['calibration'], target_axis_scale, target_label)
    _residual_scatter(axes[2], fitted, selected_errors, target_label, error_label, n)
    _qq(axes[3], errors, error_label)
    _absolute_scatter(axes[4], fitted, np.abs(selected_errors), target_label, error_label, n)
    _daily(axes[5], tables['daily'], error_axis_scale, error_label, n)
    return finish_figure(figure)


def worst_slices_figure(table, *, unit='units', title='Candidate worst slices'):
    """Show ranked slice MAE alongside support and total absolute-error contribution."""
    # PLOTTING LOGIC: Preserve table ranking and flags; shares are comparable within each slice specification.
    labels = [f'{spec}: {group}' + (' †' if low else '')
              for spec, group, low in zip(table['spec'], table['group'], table['low_support'])]
    figure = make_figure((15, row_figure_height(labels)))
    axes = figure.subplots(1, 3, gridspec_kw={'width_ratios': [3, 1, 1.4]})
    figure_title(figure, title, '† below minimum support · Shares use the full candidate population per specification')
    if table.empty:
        for axis, heading in zip(axes, ['Candidate MAE', 'Records', 'Share of absolute error']):
            _empty_axis(axis, heading, 'No configured slice groups')
        return finish_figure(figure)
    y = np.arange(len(table))
    scale = _display_scale(table['candidate_mae'])
    colors = np.where(table['low_support'], '#b1bdc5', COLOR)
    axes[0].barh(y, table['candidate_mae'] / scale, color=colors)
    axes[1].barh(y, table['n'], color=REFERENCE_COLOR)
    axes[2].barh(y, table['error_share_pct'], color=COLOR)
    for axis, heading, label in zip(axes, ['Candidate MAE', 'Records', 'Share of absolute error'],
                                    [_unit_label(unit, scale), 'N', '% of total error']):
        categorical_ticks(axis, labels if axis is axes[0] else ['']*len(labels), axis='y', width=32)
        axis.set(title=heading, xlabel=label)
        axis.margins(y=.02, x=.08)
        axis.invert_yaxis()
    return finish_figure(figure)
