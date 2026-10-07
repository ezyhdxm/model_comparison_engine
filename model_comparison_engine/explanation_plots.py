"""Portable explanation figures with complete labels and explicit prediction scales."""
# SETUP LOGIC: Figures share the review's local style without changing global plotting settings.
import numpy as np
import pandas as pd
from .plot_style import (TEAL, SLATE, ORANGE, make_figure, finish_figure, figure_title,
                         categorical_ticks, row_figure_height)


def _limit(top_n):
    # VALIDATION LOGIC: Keep display limits explicit; full feature tables remain available in the result.
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or not 1 <= top_n <= 100:
        raise ValueError('top_n must be an integer between 1 and 100.')


def plot_permutation(result, *, top_n=20):
    # PLOTTING LOGIC: Rank by the requested held-out error metric and show shuffle SD, not a confidence interval.
    _limit(top_n)
    metric = result.settings['metric']
    table = result.tables['importance'].head(top_n)
    labels = table['group'].astype(str).tolist()
    values = table[f'{metric}_increase_mean'].to_numpy(dtype=float)
    variation = table[f'{metric}_increase_std'].to_numpy(dtype=float)
    fig = make_figure((12, row_figure_height(labels, width=38)))
    ax = fig.subplots()
    ax.barh(np.arange(len(table)), values, xerr=np.where(np.isfinite(variation), variation, 0),
            color=np.where(values >= 0, TEAL, ORANGE), height=.65, capsize=3, zorder=3)
    ax.axvline(0, color=SLATE, linewidth=.8)
    categorical_ticks(ax, labels, axis='y', width=38)
    ax.invert_yaxis()
    ax.set(xlabel=f'{metric.upper()} increase after shuffling ({result.settings["unit"]})')
    baseline = result.tables['baseline'].iloc[0]
    context = ', '.join(result.settings['shuffle_within']) or 'all selected records'
    figure_title(fig, f'{result.settings["model"]} | Permutation importance',
                 f'N={baseline["n"]:,.0f} · Baseline {metric.upper()}={baseline[metric]:.4g} · '
                 f'{result.settings["repeats"]} repeats · Bars ± shuffle SD (not a CI) · Shuffle within: {context}')
    return finish_figure(fig)


def plot_shap_global(result, *, top_n=20):
    # PLOTTING LOGIC: Separate contribution size from signed direction; neither panel measures accuracy gain.
    _limit(top_n)
    table = result.tables['global_importance'].head(top_n)
    labels = table['feature'].astype(str).tolist()
    fig = make_figure((14, row_figure_height(labels, width=32)))
    magnitude, direction = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.15, 1]})
    y = np.arange(len(table))
    magnitude.barh(y, table['mean_abs_shap'], color=TEAL, height=.65, zorder=3)
    direction.barh(y, table['mean_signed_shap'], color=np.where(table['mean_signed_shap'] >= 0, ORANGE, TEAL),
                   height=.65, zorder=3)
    categorical_ticks(magnitude, labels, axis='y', width=32)
    categorical_ticks(direction, [''] * len(labels), axis='y')
    magnitude.set(title='Mean absolute contribution', xlabel=result.settings['unit'])
    direction.set(title='Mean signed contribution', xlabel=result.settings['unit'])
    direction.axvline(0, color=SLATE, linewidth=.8)
    for ax in (magnitude, direction):
        ax.invert_yaxis()
        ax.margins(x=.08)
    figure_title(fig, f'{result.settings["model"]} | Tree SHAP',
                 f'{result.settings["evaluated_rows"]:,} explained records · Contributions to the learned prediction · '
                 'Magnitude is not accuracy improvement; signed means can cancel' +
                 (' · Explicit cases only' if result.settings.get('row_selection') == 'explicit_positions' else ''))
    return finish_figure(fig)


def _local_display(result, row_position, top_n):
    # VALIDATION LOGIC: Only saved local rows can be plotted; no hidden SHAP calculation is triggered.
    local = result.tables['local_contributions']
    if local.empty:
        raise ValueError('There are no saved local explanations.')
    position = int(local.iloc[0]['row_position']) if row_position is None else row_position
    if position not in set(local['row_position']):
        raise ValueError('row_position is not in the saved local explanations; request it explicitly with tree_shap.')
    # CORE LOGIC: STEP 1 — Retain largest local contributions and sum the remaining features without losing additivity.
    # Input: contributions={'a':3,'b':-2,'c':1}, top_n=2.
    # Output: displayed contributions={'a':3,'b':-2,'Other 1 features':1}; total remains 2.
    # Explanation: Display selection uses absolute contribution; the omitted signed contributions are aggregated.
    # Trick: Summing signed values preserves reconstruction, whereas summing magnitudes would overstate the prediction.
    table = local.loc[local['row_position'].eq(position)].copy()
    order = np.argsort(-np.abs(table['shap_value_scaled'].to_numpy()), kind='stable')
    table = table.iloc[order]
    shown = table.head(top_n).copy()
    omitted = table.iloc[top_n:]
    # CORE LOGIC: STEP 2 — Append one explicit remainder row only when features were omitted.
    # Input: omitted contributions=[1,-0.5], shown has two feature rows.
    # Output: shown gains {'feature':'Other 2 features','feature_value':'','shap_value_scaled':0.5}.
    # Explanation: A zero remainder is still a real aggregate of omitted features, not an unavailable contribution.
    # Trick: The remainder uses only the plotted scale; the full native contribution table remains unchanged.
    if not omitted.empty:
        extra = dict(feature=f'Other {len(omitted)} features', feature_value='',
                     shap_value_scaled=omitted['shap_value_scaled'].sum())
        shown = pd.concat([shown, pd.DataFrame([extra])], ignore_index=True)
    return position, shown


def plot_shap_local(result, *, row_position=None, top_n=15):
    # CONFIGURATION LOGIC: Default to the first saved local case, usually the largest sampled final-target error.
    _limit(top_n)
    position, shown = _local_display(result, row_position, top_n)
    prediction = result.tables['predictions'].set_index('row_position').loc[position]
    # PLOTTING LOGIC: Horizontal signed bars keep small learned effects visible beside a large fixed offset.
    labels = [f'{row.feature} = {_display_value(row.feature_value)}' if str(row.feature_value) else str(row.feature)
              for row in shown.itertuples()]
    values = shown['shap_value_scaled'].to_numpy(dtype=float)
    fig = make_figure((13, row_figure_height(labels, width=42, minimum=5.5)))
    ax = fig.subplots()
    ax.barh(np.arange(len(shown)), values, color=np.where(values >= 0, ORANGE, TEAL), height=.65, zorder=3)
    categorical_ticks(ax, labels, axis='y', width=42)
    ax.axvline(0, color=SLATE, linewidth=.8)
    ax.invert_yaxis()
    ax.set(xlabel=f'Contribution to learned prediction ({result.settings["unit"]}); positive pushes higher')
    reconstruction = (f'Native target units: base {prediction["base_value"]:.5g} + contributions '
                      f'{prediction["shap_sum"]:.5g} + fixed offset {prediction["offset"]:.5g} '
                      f'= final prediction {prediction["final_prediction"]:.5g}')
    if 'actual' in prediction.index:
        reconstruction += f' · Actual {prediction["actual"]:.5g}'
    figure_title(fig, f'{result.settings["model"]} | Local SHAP · source position {position}', reconstruction)
    return finish_figure(fig)


def _display_value(value):
    # PLOTTING LOGIC: Keep numerical labels compact; unrounded inputs remain available in the exported table.
    return f'{value:.5g}' if isinstance(value, (float, np.floating)) else str(value)
