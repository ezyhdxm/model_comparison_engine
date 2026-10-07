"""Readable paired trade plots and optional hover-enabled Plotly views."""
# SETUP LOGIC: Static output needs only the existing Matplotlib dependency; Plotly is imported on demand.
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, DateFormatter
from matplotlib.lines import Line2D
from .plot_style import make_figure, finish_figure, figure_title, wrapped_label, TEAL, SLATE, ORANGE

# PLOTTING LOGIC: Symbols retain exact supplied side categories without inventing buy/sell conventions.
MARKERS = ['o', '^', 's', 'D', 'v', 'P', 'X']
PLOTLY_MARKERS = ['circle', 'triangle-up', 'square', 'diamond', 'triangle-down', 'cross', 'x']


def _date_axis(axis, timezone):
    # PLOTTING LOGIC: Local labels include date changes and retain DST offset in the interactive hover.
    # PLOTTING LOGIC: Two-tick minimum permits a coarser calendar unit at interval boundaries.
    # Requiring three ticks over 49 hours can force hourly ticks beyond Matplotlib's 12-hour ladder.
    locator = AutoDateLocator(minticks=2, maxticks=6, tz=ZoneInfo(timezone))
    axis.xaxis.set_major_locator(locator)
    span_seconds = np.ptp(axis.get_xlim()) * 86400
    formatter = DateFormatter('%H:%M:%S\n%Y-%m-%d', tz=ZoneInfo(timezone)) if span_seconds < 300 else ConciseDateFormatter(locator, tz=ZoneInfo(timezone))
    axis.xaxis.set_major_formatter(formatter)
    axis.set_xlabel(f'Time · {timezone}')


def _empty(axis, message):
    # PLOTTING LOGIC: Unsupported time/data populations are explicit rather than shown as zero predictions.
    axis.text(.5, .5, message, ha='center', va='center', transform=axis.transAxes, color=SLATE, wrap=True)
    axis.set_axis_off()


def intraday_figure(tables, *, reference_name='Reference', candidate_name='Candidate', unit='units'):
    """Plot full-population paired MAE, RMSE, bias and support at the configured resolution."""
    # PLOTTING LOGIC: Unsupported bins retain their metrics in tables but are masked in loss curves.
    figure = make_figure((14, 9))
    axes = figure.subplots(2, 2)
    frame, summary = tables['intraday'], tables['summary'].iloc[0]
    if frame.empty:
        for axis in axes.ravel():
            _empty(axis, 'Map a timestamp column to view intraday comparisons.')
        figure_title(figure, 'Paired comparisons through time', 'No timed paired records are available')
        return finish_figure(figure)
    for axis, metric, label in zip(axes.ravel()[:3], ['mae', 'rmse', 'bias'], ['MAE', 'RMSE', 'Signed bias']):
        for role, name, color in [('reference', reference_name, SLATE), ('candidate', candidate_name, TEAL)]:
            values = frame[f'{role}_{metric}'].where(frame['supported'])
            axis.plot(frame['time'], values, marker='.', markersize=3, linewidth=1.2, label=wrapped_label(name), color=color)
        axis.set(title=f'{label} · same paired records in every bin', ylabel=unit)
        if metric == 'bias':
            axis.axhline(0, color=ORANGE, linewidth=.8, linestyle='--')
        axis.legend(loc='best', fontsize=8)
        axis.set_xlim(frame['time'].iloc[0], frame['time_end'].iloc[-1])
        _date_axis(axis, summary['timezone'])
    axis = axes[1, 1]
    axis.plot(frame['time'], frame['n'], color=SLATE, linewidth=1, label='Paired records')
    axis.axhline(summary['min_count'], color=ORANGE, linestyle='--', label=f"Minimum N={summary['min_count']}")
    low = frame['n'].gt(0) & ~frame['supported']
    axis.scatter(frame.loc[low, 'time'], frame.loc[low, 'n'], color=ORANGE, s=16, label='Below minimum support')
    axis.set(title='Record counts · gaps retained', ylabel='Paired records', ylim=(0, None))
    axis.legend(loc='best', fontsize=8)
    axis.set_xlim(frame['time'].iloc[0], frame['time_end'].iloc[-1])
    _date_axis(axis, summary['timezone'])
    figure_title(figure, f'{candidate_name} vs {reference_name} · paired time comparison',
                 f"{summary['frequency']} fixed UTC elapsed bins, displayed in {summary['timezone']} · "
                 f"{int(summary['supported_bins']):,}/{int(summary['total_bins']):,} supported bins · "
                 'all paired records, no point sampling\nEmpty/low-support bins break the loss curves; positive bias means overprediction.')
    return finish_figure(figure)


def _metadata(points):
    # PLOTTING LOGIC: Deterministic display dictionaries share exactly the same colors and symbols across panels.
    sides = list(points['__view_side'].drop_duplicates())
    categories = list(points['__view_color'].drop_duplicates())
    markers = dict(zip(sides, MARKERS))
    palette = colormaps['tab20']
    colors = {value: palette(index % 20) for index, value in enumerate(categories)}
    return markers, colors


def _timeline(axis, points, markers, reference_name, candidate_name, timezone):
    # PLOTTING LOGIC: Scatter events independently; never connect different bonds into a fictitious price path.
    timed = points.loc[points['__time'].notna()]
    if timed.empty:
        _empty(axis, 'No timestamps in the displayed paired sample; prediction-vs-actual remains available.')
        return
    models = [('__actual', 'Actual', '#263747'), ('__reference', reference_name, SLATE), ('__candidate', candidate_name, TEAL)]
    for column, name, color in models:
        for side, marker in markers.items():
            selected = timed.loc[timed['__view_side'].eq(side)]
            axis.scatter(selected['__time'], selected[column], s=selected['__view_size'], marker=marker,
                         color=color, alpha=.55, linewidths=.35, edgecolors='white')
    handles = [Line2D([], [], linestyle='', marker='o', color=color, label=wrapped_label(name)) for _, name, color in models]
    first = axis.legend(handles=handles, loc='upper left', fontsize=8)
    axis.add_artist(first)
    side_handles = [Line2D([], [], linestyle='', marker=marker, color='#263747', label=wrapped_label(side)) for side, marker in markers.items()]
    axis.legend(handles=side_handles, loc='upper right', fontsize=8, title='Mapped side values')
    axis.set(title='Actual and both predictions · independent events', ylabel='Spread / target level · source units')
    lower, upper = timed['__time'].min(), timed['__time'].max()
    padding = (upper - lower) * .04 if upper > lower else pd.Timedelta(seconds=30)
    axis.set_xlim(lower - padding, upper + padding)
    _date_axis(axis, timezone)


def _prediction_scatter(axis, points, markers, colors, reference_name, candidate_name):
    # PLOTTING LOGIC: The identity diagonal means exact prediction; category colors only annotate candidate points.
    actual = points['__actual'].to_numpy()
    levels = points[['__actual', '__reference', '__candidate']].to_numpy(dtype=float)
    lower, upper = float(levels.min()), float(levels.max())
    axis.plot([lower, upper], [lower, upper], color=ORANGE, linestyle='--', linewidth=1, label='Prediction = actual')
    axis.scatter(actual, points['__reference'], color=SLATE, s=15, marker='+', alpha=.22,
                 label=wrapped_label(reference_name) + ' · +')
    for category, color in colors.items():
        for side, marker in markers.items():
            selected = points.loc[points['__view_color'].eq(category) & points['__view_side'].eq(side)]
            axis.scatter(selected['__actual'], selected['__candidate'], color=color, s=selected['__view_size'],
                         marker=marker, alpha=.65, edgecolors='white', linewidths=.3)
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(Line2D([], [], linestyle='', marker='o', color=color, label=wrapped_label(category)) for category, color in colors.items())
    axis.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8,
                title='Candidate color categories', borderaxespad=0)
    axis.set(title=f'{candidate_name} and {reference_name} vs actual',
             xlabel='Actual level · source units', ylabel='Prediction level · source units')


def trade_figure(tables, *, reference_name='Reference', candidate_name='Candidate', unit='units'):
    """Static source-unit predictions with explicit side, dealer/counterparty and size encodings."""
    # PLOTTING LOGIC: Sample size and missing-time omissions remain visible beside all point views.
    points, summary = tables['points'], tables['summary'].iloc[0]
    labels = points['__view_color'].drop_duplicates().tolist()
    legend_lines = sum(wrapped_label(value).count('\n') + 1 for value in labels)
    figure = make_figure((17, max(7.5, 2.8 + .18 * legend_lines)))
    axes = figure.subplots(1, 2)
    if points.empty:
        for axis in axes:
            _empty(axis, 'No finite paired records in the applied population.')
    else:
        markers, colors = _metadata(points)
        _timeline(axes[0], points, markers, reference_name, candidate_name, summary['timezone'])
        _prediction_scatter(axes[1], points, markers, colors, reference_name, candidate_name)
    color_column = tables['settings']['dealer_column'] or tables['settings']['counterparty_column'] or '(unmapped)'
    figure_title(figure, 'Paired trade predictions · source target units',
                 f"Displayed {len(points):,}/{int(summary['paired_rows']):,} paired records · seeded uniform sample when capped · "
                 f"timeline omits {int(points['__time'].isna().sum()) if len(points) else 0:,} displayed untimed records\n"
                 f"Color: {color_column}; symbol: mapped side; area increases with square-root quantity (bounded 12–140). "
                 'Unknown/nonpositive size uses minimum area; original role values are not reinterpreted.')
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
    template = '<br>'.join(f'{label}: %{{customdata[{index}]}}' for index, label in enumerate(labels))
    template += ('<br>Local time: %{customdata[6]}<br>Actual: %{customdata[7]:.6g}'
                 '<br>Reference: %{customdata[8]:.6g}<br>Candidate: %{customdata[9]:.6g}'
                 '<br>Quantity status: %{customdata[10]}<extra>%{fullData.name}</extra>')
    customdata = np.empty((len(points), len(arrays)), dtype=object)
    for index, values in enumerate(arrays):
        customdata[:, index] = values
    return customdata, template


def trade_interactive(tables, *, reference_name='Reference', candidate_name='Candidate', unit='units'):
    """Return a Plotly Figure with zoom and point metadata; raises ImportError when Plotly is absent."""
    # SETUP LOGIC: Optional interactivity never blocks static figures, tables or report exports.
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError('Install model-comparison-engine[interactive] or plotly to enable hover-enabled trade views.') from exc
    # PLOTTING LOGIC: Both panels use the identical deterministic sample and restored final-level predictions.
    points, summary = tables['points'], tables['summary'].iloc[0]
    figure = make_subplots(rows=2, cols=1, subplot_titles=('Actual and predictions through time', 'Predictions versus actual'))
    if points.empty:
        figure.add_annotation(text='No finite paired records', x=.5, y=.5, showarrow=False)
        return figure
    side_labels = list(points['__view_side'].drop_duplicates())
    symbols = points['__view_side'].map(dict(zip(side_labels, PLOTLY_MARKERS)))
    hover, template = _hover_data(points, tables['settings'], summary)
    diameter = np.sqrt(points['__view_size'].to_numpy()) * 1.5
    timed = points['__time'].notna().to_numpy()
    utc_times = pd.to_datetime(points['__time'], utc=True).dt.tz_localize(None)
    for column, name, color in [('__actual', 'Actual', '#263747'), ('__reference', reference_name, SLATE), ('__candidate', candidate_name, TEAL)]:
        figure.add_trace(go.Scattergl(x=utc_times.loc[timed], y=points.loc[timed, column], mode='markers', name=name,
                                     marker=dict(color=color, size=diameter[timed], symbol=symbols.loc[timed].tolist(), opacity=.65),
                                     customdata=hover[timed], hovertemplate=template), row=1, col=1)
    figure.add_trace(go.Scattergl(x=points['__actual'], y=points['__reference'], mode='markers', name=reference_name + ' vs actual',
                                 marker=dict(color=SLATE, size=5, symbol='cross', opacity=.25), customdata=hover,
                                 hovertemplate=template), row=2, col=1)
    _, colors = _metadata(points)
    for category, rgba in colors.items():
        selected = points['__view_color'].eq(category).to_numpy()
        color = 'rgb(%d,%d,%d)' % tuple(int(value * 255) for value in rgba[:3])
        figure.add_trace(go.Scattergl(x=points.loc[selected, '__actual'], y=points.loc[selected, '__candidate'], mode='markers',
                                     name=wrapped_label(category, 32).replace('\n', '<br>'), legendgroup='categories',
                                     marker=dict(color=color, size=diameter[selected], symbol=symbols.loc[selected].tolist(), opacity=.75),
                                     customdata=hover[selected], hovertemplate=template), row=2, col=1)
    levels = points[['__actual', '__reference', '__candidate']].to_numpy(dtype=float)
    figure.add_trace(go.Scatter(x=[levels.min(), levels.max()], y=[levels.min(), levels.max()], mode='lines',
                               name='Prediction = actual', line=dict(color=ORANGE, dash='dash'), hoverinfo='skip'), row=2, col=1)
    for side, symbol in zip(side_labels, PLOTLY_MARKERS):
        figure.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Side: ' + wrapped_label(side, 26).replace('\n', '<br>'),
                                   legendgroup='side', marker=dict(color='#263747', size=8, symbol=symbol), hoverinfo='skip'), row=1, col=1)
    # PLOTTING LOGIC: View settings state source units and sample counts; hover includes exact timestamps and original metadata.
    legend_lines = sum(str(trace.name).count('<br>') + 1 for trace in figure.data)
    figure.update_layout(template='plotly_white', height=max(900, 170 + 19 * legend_lines), hovermode='closest',
                         title=f"Paired trades: {len(points):,}/{int(summary['paired_rows']):,} displayed · source target units",
                         legend=dict(orientation='v', yanchor='top', y=1, x=1.02, groupclick='toggleitem'),
                         margin=dict(l=70, r=270, t=80, b=80))
    figure.update_xaxes(title_text=f"Time · UTC (hover also shows {summary['timezone']} and exact offset)", row=1, col=1, nticks=6)
    figure.update_xaxes(title_text='Actual level · source units', row=2, col=1, nticks=6)
    figure.update_yaxes(title_text='Spread / target level · source units', row=1, col=1)
    figure.update_yaxes(title_text='Prediction level · source units', row=2, col=1)
    return figure
