"""Significance heatmaps with signed t statistics and an explicit support audit."""
# SETUP LOGIC: Unmanaged figures work in notebooks and headless report export.
import numpy as np
from .plot_style import (INK, make_figure, finish_figure, figure_title, categorical_ticks,
                         heatmap_size, wrapped_label)


def _plot_coordinates(table):
    # PLOTTING LOGIC: Present global, one-way, and interaction tables in one aligned grid.
    if {'x', 'y'}.issubset(table.columns):
        return table.assign(_row=table['x'], _column=table['y'])
    if 'group' in table.columns:
        return table.assign(_row=table['group'], _column='Paired comparison')
    if len(table) > 1:
        raise ValueError('Significance tables need group or x/y columns, or a single overall row.')
    return table.assign(_row='Overall', _column='Paired comparison')


def _cell_label(record):
    # PLOTTING LOGIC: Stars encode adjusted-p thresholds only; support flags use words.
    if record['status'] == 'ok' and np.isfinite(record['t_statistic']):
        return f't={record["t_statistic"]:.3g}{record["significance"]}\nq={record["q_value"]:.3g}'
    status = str(record['status']).replace('_', ' ')
    return 'Untested\n'+wrapped_label(status, 20)


def significance_heatmap(table, title='Paired loss significance', unit_label='units'):
    """Display t/q and unit/row support; blank combinations have no observations."""
    # PLOTTING LOGIC: An empty slice still exports a useful, explicit figure.
    if table.empty:
        figure = make_figure((10, 4))
        axis = figure.subplots()
        axis.text(.5, .5, 'No observed groups', ha='center', va='center')
        axis.set_axis_off()
        figure_title(figure, title)
        return finish_figure(figure)
    # CORE LOGIC: STEP 1 — Align statistic, support, and row-reference grids to identical category order.
    # Input: (x,y,t,unit_count)=[('A','big',-2,10),('A','small',1,12),('B','big',3,11)].
    # Output: t grid=[[-2,1],[3,NaN]], support grid=[[10,12],[11,NaN]], row grid=[[0,1],[2,NaN]].
    # Explanation: B/small has no record in the input table and stays missing in every aligned grid.
    # Trick: Explicit reindexing preserves displayed category order; absent combinations remain blank.
    cells = _plot_coordinates(table).reset_index(drop=True).assign(_position=np.arange(len(table)))
    xs, ys = cells['_row'].drop_duplicates().tolist(), cells['_column'].drop_duplicates().tolist()
    grids = {name: cells.pivot(index='_row', columns='_column', values=name).reindex(index=xs, columns=ys)
             for name in ['t_statistic', 'unit_count', '_position']}
    values = grids['t_statistic'].to_numpy(dtype=float)
    counts = grids['unit_count'].to_numpy(dtype=float)
    # PLOTTING LOGIC: Negative t favors the candidate; the color scale is centered on zero.
    stacked = len(ys) > 8
    figure = make_figure(heatmap_size(xs, ys, cell_width=1.3, cell_height=.82, column_width=18, stacked=stacked))
    axes = figure.subplots(2, 1) if stacked else figure.subplots(1, 2)
    bound = max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0, 1e-9)
    statistic_image = axes[0].imshow(np.ma.masked_invalid(values), cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
    count_image = axes[1].imshow(np.ma.masked_invalid(counts), cmap='Blues', vmin=0, vmax=max(float(np.nanmax(counts)), 1), aspect='auto')
    for axis, image, label in zip(axes, [statistic_image, count_image], ['Two-sided paired loss t statistic', 'Inference-unit support']):
        categorical_ticks(axis, ys, width=18)
        categorical_ticks(axis, xs, axis='y', width=28)
        axis.set_title(label)
        figure.colorbar(image, ax=axis, shrink=.65, fraction=.025, pad=.025)
    # PLOTTING LOGIC: Untested observed cells show their reason; absent combinations stay entirely blank.
    for i in range(len(xs)):
        for j in range(len(ys)):
            position = grids['_position'].iloc[i, j]
            if not np.isfinite(position):
                continue
            record = cells.iloc[int(position)]
            ink = 'white' if np.isfinite(values[i, j]) and abs(values[i, j]) > .55*bound else INK
            axes[0].text(j, i, _cell_label(record), ha='center', va='center', fontsize=8, color=ink)
            count = int(record['unit_count'])
            unit = str(record.get('inference_unit', 'record'))
            noun = {'record': 'records', 'date': 'dates', 'entity': 'entities'}.get(unit, 'units')
            noun = unit if count == 1 and unit in {'record', 'date', 'entity'} else noun
            support = f'{count:,} {noun}\n{int(record["tested_n"]):,}/{int(record["paired_n"]):,} rows used'
            if record['low_support']:
                support += '\nLOW SUPPORT'
            if record['excluded_unit_rows']:
                support += f'\n{int(record["excluded_unit_rows"]):,} missing unit'
            axes[1].text(j, i, support, ha='center', va='center', fontsize=8,
                         color='white' if count_image.norm(counts[i, j]) > .55 else INK)
    # PLOTTING LOGIC: State the correction family, independent-unit assumption, and pointwise-CI boundary.
    first = cells.iloc[0]
    correction = str(first.get('correction', 'by')).upper()
    inference_unit = str(first.get('inference_unit', 'record'))
    loss = str(first.get('loss', 'absolute'))
    loss_units = unit_label+'²' if loss == 'squared' else unit_label
    alpha = float(first.get('alpha', .05))
    figure_title(figure, title+f' | {loss} error loss ({loss_units}) | {inference_unit} units',
                   f'{correction} correction within this table; alpha={alpha:g}; table CIs: {1-alpha:.1%} pointwise, unadjusted\n'
                   'Negative t favors candidate; stars: q ≤ .001 ***, q ≤ .01 **, q ≤ .05 *; blank = no records\n'
                   'Units must be independent; date/entity means are equally weighted. Serial dependence and repeated exploration remain.')
    return finish_figure(figure)
