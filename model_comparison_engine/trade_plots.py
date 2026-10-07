"""Compact paired trade plots with categorical colors and optional hover-enabled views."""
# SETUP LOGIC: Static figures use Matplotlib; Plotly is imported only on an explicit request.
from colorsys import hls_to_rgb
from hashlib import sha256
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.dates import (AutoDateLocator, DateFormatter, ConciseDateFormatter,
                              HOURLY, DAILY, MINUTELY, SECONDLY, MONTHLY)
from matplotlib.lines import Line2D
from .plot_style import make_figure, finish_figure, figure_title, wrapped_label, SLATE

# PLOTTING LOGIC: D/B/S colors preserve literal supplied codes, without inferring their economic meaning.
SIDE_COLORS = {'D': '#7B4AB5', 'B': '#2478B5', 'S': '#D66B20'}
REFERENCE_COLOR, CANDIDATE_COLOR = '#2563A6', '#CC5B27'
RESERVED_LABELS = {'Unmapped', 'Not mapped', 'Missing (not supplied)', 'Other categories (display only)'}


class _FractionalDateFormatter(DateFormatter):
    # PLOTTING LOGIC: Keep the DateFormatter type so shared figure styling preserves these temporal coordinates.
    def __init__(self, timezone, digits):
        super().__init__('%H:%M:%S.%f\n%d %b', tz=ZoneInfo(timezone))
        self.digits = digits

    def __call__(self, value, position=0):
        # PLOTTING LOGIC: Milliseconds are sufficient for broad subsecond ranges; finer views retain microseconds.
        clock, date = super().__call__(value, position).split('\n')
        return clock[:9 + self.digits] + '\n' + date


def _date_axis(axis, timezone):
    # PLOTTING LOGIC: Permit enough intervals at clock boundaries to avoid fallback warnings and sparse month-only ticks.
    span_seconds = np.ptp(axis.get_xlim()) * 86400
    locator = AutoDateLocator(minticks=2 if span_seconds < 1 else 4, maxticks=4 if span_seconds < 1 else 6,
                              interval_multiples=False, tz=ZoneInfo(timezone))
    locator.intervald[HOURLY] = [1, 2, 3, 4, 6, 12, 24]
    locator.intervald[DAILY] = [1, 2, 3, 7, 14, 21, 28]
    locator.intervald[MINUTELY] = locator.intervald[SECONDLY] = [1, 5, 10, 15, 30, 60]
    locator.intervald[MONTHLY] = [1, 2, 3, 4, 6, 12]
    axis.xaxis.set_major_locator(locator)
    if span_seconds < 1:
        formatter = _FractionalDateFormatter(timezone, 3 if span_seconds >= .01 else 6)
    elif span_seconds >= 10 * 86400:
        formatter = DateFormatter('%d %b\n%Y' if span_seconds >= 365 * 86400 else '%d %b', tz=ZoneInfo(timezone))
    elif span_seconds < 300:
        formatter = DateFormatter('%H:%M:%S\n%d %b', tz=ZoneInfo(timezone))
    else:
        formatter = ConciseDateFormatter(locator, tz=ZoneInfo(timezone))
    axis.xaxis.set_major_formatter(formatter)
    axis.set_xlabel(f'Time · {timezone}')


def _empty(axis, message):
    # PLOTTING LOGIC: Unavailable values are never shown as zero predictions or errors.
    axis.text(.5, .5, message, ha='center', va='center', transform=axis.transAxes, color=SLATE, wrap=True)
    axis.set_axis_off()


def intraday_figure(tables, *, reference_name='Reference', candidate_name='Candidate', unit='units'):
    """Render full-population paired losses and counts with shared, distinct model styles."""
    # PLOTTING LOGIC: Mask unsupported plotted errors while preserving their original table values and time gaps.
    figure = make_figure((12.5, 7.2))
    axes = figure.subplots(2, 2)
    frame, summary = tables['intraday'], tables['summary'].iloc[0]
    if frame.empty:
        for axis in axes.ravel():
            _empty(axis, 'No timed paired records')
        figure_title(figure, 'Paired errors through time', 'Map a timestamp to see intraday comparisons.')
        return finish_figure(figure)
    models = [('reference', REFERENCE_COLOR, '--'), ('candidate', CANDIDATE_COLOR, '-')]
    for axis, metric, label in zip(axes.ravel()[:3], ['mae', 'rmse', 'bias'], ['MAE', 'RMSE', 'Signed bias']):
        for role, color, style in models:
            values = frame[f'{role}_{metric}'].where(frame['supported'])
            axis.plot(frame['time'], values, marker='o', markersize=2.5, linewidth=1.6,
                      linestyle=style, color=color, alpha=.95)
        axis.set(title=label, ylabel=unit)
        if metric == 'bias':
            axis.axhline(0, color='#718096', linewidth=.7, linestyle=':')
        axis.set_xlim(frame['time'].iloc[0], frame['time_end'].iloc[-1])
        _date_axis(axis, summary['timezone'])
    # PLOTTING LOGIC: Bars occupy their elapsed bin widths; zero-count bins remain empty spaces, not line plunges.
    count_axis = axes[1, 1]
    width_days = (frame['time_end'] - frame['time']).dt.total_seconds().to_numpy() / 86400 * .88
    count_colors = np.where(frame['supported'], '#80A9C8', '#D59957')
    count_axis.bar(frame['time'], frame['n'], width=width_days, align='edge', color=count_colors, linewidth=0)
    count_axis.axhline(summary['min_count'], color='#72513A', linestyle=':', linewidth=1)
    count_axis.set(title=f"Trade count · support N ≥ {int(summary['min_count']):,}", ylabel='Paired records', ylim=(0, None))
    count_axis.set_xlim(frame['time'].iloc[0], frame['time_end'].iloc[-1])
    _date_axis(count_axis, summary['timezone'])
    # PLOTTING LOGIC: A dedicated short header row holds the only model legend, outside data areas.
    names = [(reference_name, REFERENCE_COLOR, '--'), (candidate_name, CANDIDATE_COLOR, '-')]
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=2, label=wrapped_label(name, 38)) for name, color, style in names]
    axes[0, 0].legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.10), ncol=2, borderaxespad=0,
                      fontsize=8, handlelength=3, columnspacing=1.8)
    figure_title(figure, 'Paired errors through time',
                 f"{summary['frequency']} bins · {int(summary['supported_bins']):,}/{int(summary['total_bins']):,} supported · "
                 f"{int(summary['empty_bins']):,} empty · all paired records\n"
                 'Gaps are retained. Positive bias means overprediction; tan count bars fall below minimum support.')
    return finish_figure(figure)


def _category_label(value):
    # PLOTTING LOGIC: Remove only the adapter's display prefix; retain literal category content and explicit missing labels.
    text = str(value)
    literal = text[7:] if text.startswith('Value: ') else None
    return literal if literal is not None and literal not in RESERVED_LABELS else text


def _category_color(value):
    # PLOTTING LOGIC: Colors depend on the literal category, so filtering, sampling and order cannot recolor a code.
    text = _category_label(value)
    if text in SIDE_COLORS:
        return SIDE_COLORS[text]
    if str(value) in {'Unmapped', 'Not mapped', 'Missing (not supplied)'}:
        return '#8A96A3'
    if str(value) == 'Other categories (display only)':
        return '#5F6C79'
    hue = int.from_bytes(sha256(text.encode('utf-8')).digest()[:4], 'big') / 2**32
    return to_hex(hls_to_rgb(hue, .43, .58))


def _metadata(points):
    # PLOTTING LOGIC: Canonical codes lead the shared legend; all remaining category labels sort deterministically.
    categories = list(points['__view_color'].drop_duplicates())
    priority = {'D': 0, 'B': 1, 'S': 2}
    ordered = sorted(categories, key=lambda value: (priority.get(_category_label(value), 3), str(value)))
    return {value: _category_color(value) for value in ordered}


def _view_columns(points):
    # CONFIGURATION LOGIC: Prepared view fields contain the engine's chosen transformation; legacy tables retain levels.
    return {role: '__view_' + role if '__view_' + role in points else '__' + role
            for role in ['actual', 'reference', 'candidate']}


def _view_labels(summary):
    # PLOTTING LOGIC: Name the supplied target and distinguish source levels from scaled diagnostic deviations.
    mode = summary.get('point_view', 'level')
    target, unit = str(summary.get('actual_column', 'target')), str(summary.get('error_unit', 'units'))
    if mode == 'residual':
        return dict(mode=mode, title=f'{target}: prediction errors', time_suffix='error through time',
                    scatter_suffix='error vs actual', time_y=f'Prediction − {target} · {unit}',
                    scatter_x=f'Actual {target} · source units', scatter_y=f'Prediction − {target} · {unit}')
    if mode == 'within_entity':
        return dict(mode=mode, title=f'{target}: within-entity deviations', time_suffix='deviations through time',
                    scatter_suffix='predicted vs actual deviation', time_y=f'{target} − entity mean · {unit}',
                    scatter_x=f'Actual deviation · {unit}', scatter_y=f'Predicted deviation · {unit}')
    return dict(mode=mode, title=f'{target}: predictions and observed levels', time_suffix='vs actual through time',
                scatter_suffix='vs actual', time_y=f'{target} · source units',
                scatter_x=f'Actual {target} · source units', scatter_y=f'Predicted {target} · source units')


def _plot_points(tables):
    # PLOTTING LOGIC: Undrawable diagnostic coordinates stay in exported tables and coverage, never on a fabricated zero line.
    points = tables['points']
    return points.loc[points['__view_valid'].fillna(False)] if '__view_valid' in points else points


def _view_caption(tables, points, summary):
    # PLOTTING LOGIC: Report valid rendered rows separately from the original sample and full paired population.
    sampled = len(tables['points'])
    untimed = int(points['__time'].isna().sum()) if len(points) else 0
    unavailable = int(summary.get('missing_view_rows', 0))
    text = (f"{len(points):,} plotted / {sampled:,} sampled / {int(summary['paired_rows']):,} paired records · "
            f"{unavailable:,} paired records lack view coordinates · {untimed:,} plotted records lack time")
    note = str(summary.get('view_note', 'Levels retain source target units.'))
    focus = summary.get('focus_entity')
    if focus is not None and pd.notna(focus):
        text = f'Entity focus: {focus} · ' + text
    return text + '\n' + note


def _range_limits(values):
    # PLOTTING LOGIC: Preserve every finite plotted value with modest padding, including all visible outliers.
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return -1., 1.
    lower, upper = float(finite.min()), float(finite.max())
    padding = max(abs(lower), 1) * .01 if lower == upper else (upper - lower) * .04
    return lower - padding, upper + padding


def _level_limits(points):
    # PLOTTING LOGIC: View values already contain residuals, centered deviations or original levels; never transform twice.
    return _range_limits(points[list(_view_columns(points).values())].to_numpy(dtype=float))


def _timeline(axis, points, colors, prediction, title, timezone, limits, labels):
    # PLOTTING LOGIC: Model/actual roles use distinct markers, while every category retains one color across all panels.
    timed = points.loc[points['__time'].notna()]
    if timed.empty:
        _empty(axis, 'No timestamps in the displayed sample\nThe lower comparison panels remain available.')
        return
    actual = _view_columns(points)['actual']
    for category, color in colors.items():
        selected = timed.loc[timed['__view_color'].eq(category)]
        if labels['mode'] != 'residual':
            axis.scatter(selected['__time'], selected[actual], s=selected['__view_size'], marker='o',
                         facecolors='none', edgecolors=color, alpha=.65, linewidths=.7)
        axis.scatter(selected['__time'], selected[prediction], s=selected['__view_size'], marker='x',
                     color=color, alpha=.85, linewidths=.85)
    if labels['mode'] == 'residual':
        axis.axhline(0, color='#75808B', linestyle='--', linewidth=.8)
    axis.set(title=title, ylabel=labels['time_y'], ylim=limits)
    lower, upper = timed['__time'].min(), timed['__time'].max()
    padding = (upper - lower) * .035 if upper > lower else pd.Timedelta(seconds=30)
    axis.set_xlim(lower - padding, upper + padding)
    _date_axis(axis, timezone)


def _prediction_scatter(axis, points, colors, prediction, title, limits, labels):
    # PLOTTING LOGIC: The exact-prediction diagonal and shared scales make deviations comparable between models.
    residual = labels['mode'] == 'residual'
    actual = '__actual' if residual else _view_columns(points)['actual']
    x_limits = _range_limits(points[actual]) if residual else limits
    if residual:
        axis.axhline(0, color='#75808B', linestyle='--', linewidth=.8, alpha=.8)
    else:
        axis.plot(limits, limits, color='#75808B', linestyle='--', linewidth=.8, alpha=.8)
    for category, color in colors.items():
        selected = points.loc[points['__view_color'].eq(category)]
        axis.scatter(selected[actual], selected[prediction], color=color, s=selected['__view_size'],
                     marker='x', alpha=.8, linewidths=.85)
    axis.set(title=title, xlabel=labels['scatter_x'], ylabel=labels['scatter_y'], xlim=x_limits, ylim=limits)


def _quantity_keys(points, settings):
    # PLOTTING LOGIC: Legend examples use the same maximum and area formula as the already prepared plotted points.
    column = settings.get('quantity_column')
    if column is None or points.empty:
        return []
    values = pd.to_numeric(points[column], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    positive = values[np.isfinite(values) & (values > 0)]
    if not len(positive):
        return []
    maximum = float(positive.max())
    return [(maximum * fraction, 12 + 128 * np.sqrt(fraction)) for fraction in (.01, .25, 1)]


def _quantity_label(value):
    # PLOTTING LOGIC: Compact labels state native quantity magnitude; they never convert the underlying records.
    if value >= 1e6:
        return f'{value / 1e6:.3g}M'
    if value >= 1e3:
        return f'{value / 1e3:.3g}K'
    return f'{value:.3g}'


def _shared_legend(axis, colors, points, settings, summary):
    # PLOTTING LOGIC: A separate sidebar combines colors, actual/prediction roles and truthful quantity examples.
    axis.set_axis_off()
    column = summary.get('color_column') or settings.get('side_column') or settings.get('dealer_column') or settings.get('counterparty_column')
    heading = f'Color: {column}' if column else 'Color: metadata not mapped'
    handles = [Line2D([], [], linestyle='', color='none', label=wrapped_label(heading, 22))]
    handles += [Line2D([], [], marker='o', linestyle='', color=color, markersize=7,
                       label=wrapped_label(_category_label(category), 22)) for category, color in colors.items()]
    handles.append(Line2D([], [], color='none', label='\nMarker role'))
    if summary.get('point_view', 'level') != 'residual':
        handles.append(Line2D([], [], marker='o', markerfacecolor='none', color='#425263', linestyle='', label='Actual'))
    prediction_label = 'Prediction error' if summary.get('point_view') == 'residual' else 'Prediction'
    handles += [Line2D([], [], marker='x', color='#425263', linestyle='', label=prediction_label),
                Line2D([], [], linestyle='--', color='#75808B', label='Zero error' if summary.get('point_view') == 'residual' else 'Prediction = actual')]
    keys = _quantity_keys(points, settings)
    if settings.get('quantity_column'):
        handles.append(Line2D([], [], color='none', label='\nQuantity · bounded size'))
        handles.extend(Line2D([], [], marker='o', color='#718096', markerfacecolor='none', linestyle='',
                              markersize=np.sqrt(area), label=_quantity_label(quantity)) for quantity, area in keys)
        handles.append(Line2D([], [], marker='o', color='#718096', markerfacecolor='none', linestyle='',
                              markersize=np.sqrt(12), label='Unknown / nonpositive'))
    axis.legend(handles=handles, loc='upper left', frameon=False, borderaxespad=0, fontsize=8,
                labelspacing=.75, handletextpad=.8, handlelength=1.8)


def trade_figure(tables, *, reference_name='Reference', candidate_name='Candidate', unit='units'):
    """Render the requested residual, within-entity or level view without recomputing its transformation."""
    # PLOTTING LOGIC: Four aligned panels avoid overlaying three differently identified model series at once.
    points, summary = _plot_points(tables), tables['summary'].iloc[0]
    labels = _view_labels(summary)
    colors = _metadata(points)
    legend_lines = sum(wrapped_label(_category_label(value), 22).count('\n') + 1 for value in colors) + 15
    figure = make_figure((14, max(8.2, 1.5 + .19 * legend_lines)))
    grid = figure.add_gridspec(2, 3, width_ratios=[1, 1, .34])
    axes = [figure.add_subplot(grid[row, column]) for row, column in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    legend_axis = figure.add_subplot(grid[:, 2])
    if points.empty:
        for axis in axes:
            _empty(axis, 'No finite paired coordinates for this view')
        legend_axis.set_axis_off()
    else:
        limits = _level_limits(points)
        columns = _view_columns(points)
        for column, (role, name) in enumerate([('candidate', candidate_name), ('reference', reference_name)]):
            _timeline(axes[column], points, colors, columns[role], f"{name} · {labels['time_suffix']}", summary['timezone'], limits, labels)
            _prediction_scatter(axes[column + 2], points, colors, columns[role], f"{name} · {labels['scatter_suffix']}", limits, labels)
        _shared_legend(legend_axis, colors, tables['points'], tables['settings'], summary)
    figure_title(figure, labels['title'], _view_caption(tables, points, summary))
    return finish_figure(figure)


def _hover_data(points, settings, summary):
    # PRESENTATION LOGIC: Hover retains original role values even when rare display categories are pooled.
    columns = [summary['id_column'], summary['entity_column'], settings['side_column'], settings['counterparty_column'],
               settings['dealer_column'], settings['quantity_column']]
    labels = ['Record ID', 'Entity', 'Side', 'Counterparty', 'Dealer', 'Quantity']
    arrays = [points[column].astype('string').fillna('(missing)').to_numpy(dtype=str)
              if column else np.full(len(points), '(unmapped)') for column in columns]
    arrays.extend([points['__time'].astype('string').fillna('(missing)').to_numpy(dtype=str),
                   points['__actual'].to_numpy(), points['__reference'].to_numpy(), points['__candidate'].to_numpy(),
                   points['__view_quantity_status'].to_numpy()])
    arrays.extend([points['__error_reference'].to_numpy(), points['__error_candidate'].to_numpy()])
    view_columns = _view_columns(points)
    arrays.extend(points[view_columns[role]].to_numpy() for role in ['actual', 'reference', 'candidate'])
    template = '<br>'.join(f'{label}: %{{customdata[{index}]}}' for index, label in enumerate(labels))
    template += ('<br>Local time: %{customdata[6]}<br>Actual (source units): %{customdata[7]}'
                 '<br>Reference (source units): %{customdata[8]}<br>Candidate (source units): %{customdata[9]}'
                 '<br>Quantity status: %{customdata[10]}')
    error_unit = str(summary.get('error_unit', 'units'))
    template += f'<br>Reference error ({error_unit}): %{{customdata[11]}}<br>Candidate error ({error_unit}): %{{customdata[12]}}'
    if summary.get('point_view') == 'within_entity':
        template += (f'<br>Actual centered ({error_unit}): %{{customdata[13]}}'
                     f'<br>Reference centered ({error_unit}): %{{customdata[14]}}'
                     f'<br>Candidate centered ({error_unit}): %{{customdata[15]}}')
    template += '<extra>%{fullData.name}</extra>'
    customdata = np.empty((len(points), len(arrays)), dtype=object)
    for index, values in enumerate(arrays):
        customdata[:, index] = values
    return customdata, template


def _interactive_trace(go, points, category, color, prediction, hover, template, *, timed, x_column=None, showlegend=False):
    # PLOTTING LOGIC: Each literal category uses one color; marker shape encodes model/actual role, never side.
    selected = points['__view_color'].eq(category).to_numpy()
    selected &= points['__time'].notna().to_numpy() if timed else True
    columns = _view_columns(points)
    x = (pd.to_datetime(points['__time'], utc=True).dt.tz_localize(None) if timed else points[x_column or columns['actual']])
    actual = prediction == columns['actual']
    role_name = 'Actual' if actual else 'Prediction'
    return go.Scattergl(x=x.loc[selected], y=points.loc[selected, prediction], mode='markers',
                        name=_category_label(category) + ' · ' + role_name, legendgroup=str(category), showlegend=showlegend,
                        marker=dict(color=color, size=np.sqrt(points.loc[selected, '__view_size']) * 1.5,
                                    symbol='circle-open' if actual else 'x', opacity=.70 if actual else .85),
                        customdata=hover[selected], hovertemplate=template)


def trade_interactive(tables, *, reference_name='Reference', candidate_name='Candidate', unit='units'):
    """Return matched four-panel hover-enabled trade plots; Plotly remains optional."""
    # SETUP LOGIC: Static plots and tables remain available when optional interactivity is not installed.
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError('Install model-comparison-engine[interactive] or plotly to enable hover-enabled trade views.') from exc
    # PLOTTING LOGIC: Facets identify each model while shared categorical colors remain fixed across the whole figure.
    points, summary = _plot_points(tables), tables['summary'].iloc[0]
    labels = _view_labels(summary)
    titles = [f"{candidate_name} · {labels['time_suffix']}", f"{reference_name} · {labels['time_suffix']}",
              f"{candidate_name} · {labels['scatter_suffix']}", f"{reference_name} · {labels['scatter_suffix']}"]
    figure = make_subplots(rows=2, cols=2, subplot_titles=[wrapped_label(title, 40).replace('\n', '<br>') for title in titles],
                          horizontal_spacing=.12, vertical_spacing=.20)
    if points.empty:
        figure.add_annotation(text='No finite paired coordinates for this view', x=.5, y=.5, showarrow=False)
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        figure.update_layout(template='plotly_white', title=labels['title'], height=650, margin=dict(b=150))
        caption = wrapped_label(_view_caption(tables, points, summary), 120).replace('\n', '<br>')
        figure.add_annotation(text=caption, x=0, y=-.12, xref='paper', yref='paper',
                              xanchor='left', yanchor='top', showarrow=False, align='left', font=dict(size=11))
        return figure
    colors = _metadata(points)
    hover, template = _hover_data(points, tables['settings'], summary)
    limits = _level_limits(points)
    columns = _view_columns(points)
    residual = labels['mode'] == 'residual'
    x_column = '__actual' if residual else columns['actual']
    x_limits = _range_limits(points[x_column]) if residual else limits
    for column, role in enumerate(['candidate', 'reference'], start=1):
        for category, color in colors.items():
            timeline_columns = [columns[role]] if residual else [columns['actual'], columns[role]]
            for prediction in timeline_columns:
                trace = _interactive_trace(go, points, category, color, prediction, hover, template, timed=True)
                figure.add_trace(trace, row=1, col=column)
            trace = _interactive_trace(go, points, category, color, columns[role], hover, template, timed=False,
                                       x_column=x_column, showlegend=column == 1)
            trace.name = wrapped_label(_category_label(category), 25).replace('\n', '<br>')
            figure.add_trace(trace, row=2, col=column)
        figure.add_trace(go.Scatter(x=x_limits, y=[0, 0] if residual else limits, mode='lines', showlegend=False,
                                   line=dict(color='#75808B', dash='dash', width=1), hoverinfo='skip'), row=2, col=column)
        if residual:
            figure.add_hline(y=0, line_dash='dash', line_color='#75808B', line_width=1, row=1, col=column)
        figure.update_xaxes(title_text='Time · UTC (local offset in hover)', row=1, col=column, nticks=6)
        figure.update_yaxes(title_text=labels['time_y'], range=limits, row=1, col=column)
        figure.update_xaxes(title_text=labels['scatter_x'], range=x_limits, row=2, col=column, nticks=6)
        figure.update_yaxes(title_text=labels['scatter_y'], range=limits, row=2, col=column)
        if not points['__time'].notna().any():
            figure.update_xaxes(visible=False, row=1, col=column)
            figure.update_yaxes(visible=False, row=1, col=column)
            figure.add_annotation(text='No timed records in the displayed sample', x=.5, y=.5,
                                  xref='x domain', yref='y domain', showarrow=False, row=1, col=column)
    _interactive_legend(figure, go, tables['points'], tables['settings'], residual=residual)
    # PLOTTING LOGIC: Long legends grow vertically; plots never hide tails to create extra whitespace.
    legend_lines = sum(wrapped_label(_category_label(value), 25).count('\n') + 1 for value in colors) + 12
    color_column = summary.get('color_column') or '(not mapped)'
    figure.update_layout(template='plotly_white', height=max(820, 150 + 22 * legend_lines), hovermode='closest',
                         title=labels['title'],
                         legend=dict(orientation='v', yanchor='top', y=1, x=1.02, groupclick='togglegroup',
                                     title=dict(text='Color: ' + str(color_column))),
                         margin=dict(l=75, r=225, t=100, b=150))
    caption = wrapped_label(_view_caption(tables, points, summary), 140).replace('\n', '<br>')
    figure.add_annotation(text=caption, x=0, y=-.12, xref='paper', yref='paper',
                          xanchor='left', yanchor='top', showarrow=False, align='left', font=dict(size=11, color=SLATE))
    return figure


def _interactive_legend(figure, go, points, settings, *, residual=False):
    # PLOTTING LOGIC: Legend-only traces describe role and quantity encodings without adding fictitious data points.
    roles = [('Prediction error', 'x')] if residual else [('Actual', 'circle-open'), ('Prediction', 'x')]
    for label, symbol in roles:
        figure.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=label, legendgroup='roles',
                                   marker=dict(color='#425263', size=9, symbol=symbol), hoverinfo='skip'), row=1, col=1)
    for quantity, area in _quantity_keys(points, settings):
        figure.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Quantity ' + _quantity_label(quantity),
                                   legendgroup='quantity', marker=dict(color='#718096', size=np.sqrt(area) * 1.5,
                                   symbol='circle-open'), hoverinfo='skip'), row=1, col=1)
    if settings.get('quantity_column'):
        figure.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Unknown / nonpositive quantity',
                                   legendgroup='quantity', marker=dict(color='#718096', size=np.sqrt(12) * 1.5,
                                   symbol='circle-open'), hoverinfo='skip'), row=1, col=1)
