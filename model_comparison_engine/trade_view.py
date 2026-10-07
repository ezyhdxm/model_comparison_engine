"""Paired trade views and fixed-clock comparison bins; no prediction or fitting calls."""
# SETUP LOGIC: Reuse the existing finite paired population and stable metric definitions.
from dataclasses import asdict, dataclass
from numbers import Integral
import numpy as np
import pandas as pd
from .metrics import grouped_metrics
from .temporal import _bin_nanoseconds, _bounded_grid, _time_values, TemporalConfig


@dataclass(frozen=True)
class TradeViewConfig:
    """Explicit metadata roles; values retain the caller's meaning without buy/sell inference."""
    # CONFIGURATION LOGIC: Sampling affects plots only; every paired record enters the bin metrics.
    frequency: str = 'auto'
    min_count: int = 30
    max_points: int = 2000
    max_bins: int = 20000
    side_column: str | None = None
    counterparty_column: str | None = None
    dealer_column: str | None = None
    quantity_column: str | None = None
    max_categories: int = 12
    random_state: int = 0
    point_view: str = 'residual'
    focus_entity: object = None

    def __post_init__(self):
        # VALIDATION LOGIC: Bound displays and reject non-fixed clock intervals before allocation.
        if self.frequency != 'auto':
            _bin_nanoseconds(self.frequency)
        if self.point_view not in {'residual', 'within_entity', 'level'}:
            raise ValueError("point_view must be 'residual', 'within_entity', or 'level'.")
        if self.focus_entity is not None and (not pd.api.types.is_scalar(self.focus_entity) or pd.isna(self.focus_entity)):
            raise ValueError('focus_entity must be one nonmissing entity ID or None for all entities.')
        for name in ('min_count', 'max_points', 'max_bins', 'max_categories'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.max_categories > 18 or self.max_points > 100000:
            raise ValueError('Use max_categories <= 18 and max_points <= 100000 for bounded trade plots.')
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral) or self.random_state < 0:
            raise ValueError('random_state must be a nonnegative integer.')


def _frequency(times, timezone, requested):
    # CORE LOGIC: STEP 1 — Choose resolution from the elapsed local calendar span, not just occupied days.
    # Input: times=['2026-01-05 14:00Z','2026-01-06 15:00Z'], timezone='America/New_York', requested='auto'.
    # Output: frequency='30min', local_dates=2, calendar_span_days=2.
    # Explanation: A two-calendar-day view supports intraday bins; 4–14 days use hours and longer spans use days.
    # Trick: Two occupied days a year apart choose daily bins, preventing a misleading million-bin intraday grid.
    local = times.dropna().dt.tz_convert(timezone)
    dates = local.dt.date
    span = (dates.max() - dates.min()).days + 1 if len(dates) else 0
    automatic = '30min' if span <= 3 else ('1h' if span <= 14 else '1D')
    return automatic if requested == 'auto' else requested, int(dates.nunique()), span


def _comparison_bins(population, times, config, frequency, tolerance, timezone):
    # CORE LOGIC: STEP 1 — Map finite timed pairs onto one fixed-width UTC grid.
    # Input: times=['2026-01-01 10:05Z','2026-01-01 11:00Z'], frequency='30min'.
    # Output: bin=[0,2], grid=['10:00Z','10:30Z','11:00Z']; 10:30 has no observations.
    # Explanation: The exact 11:00 boundary begins the third left-closed bin.
    # Trick: Position arrays preserve alignment independently of source DataFrame index labels.
    known = times.notna()
    available = times.loc[known]
    settings = TemporalConfig(frequency=frequency, max_bins=config.max_bins)
    grid, bin_ns = _bounded_grid(available, settings)
    positions = [(int(value) - grid[0].value) // bin_ns for value in available.astype('int64')]
    rows = population.loc[known].assign(bin=positions)
    # CORE LOGIC: STEP 2 — Recompute paired metrics using all records before adding explicit empty bins.
    # Input: bin=[0,0,2], reference errors=[-1,3,2], candidate errors=[0,2,1], min_count=2.
    # Output: n=[2,0,1], reference_mae=[2,NaN,2], candidate_mae=[1,NaN,1], supported=[True,False,False].
    # Explanation: Plot sampling never changes these metrics; a missing bin cannot appear as a zero error.
    # Trick: Reindex after aggregation adds time gaps while preserving undefined losses and zero record counts.
    result = grouped_metrics(rows, ['bin'], tolerance=tolerance, min_count=config.min_count)
    result = result.set_index('bin').reindex(range(len(grid)))
    result[['n', 'entities', 'dates']] = result[['n', 'entities', 'dates']].fillna(0).astype('int64')
    result['supported'] = result['n'].ge(config.min_count)
    result['low_support'] = ~result['supported']
    result.insert(0, 'time', grid.tz_convert(timezone))
    result.insert(1, 'time_end', (grid + pd.Timedelta(bin_ns, unit='ns')).tz_convert(timezone))
    return result.reset_index(drop=True)


def _labels(values, limit):
    # CORE LOGIC: STEP 1 — Retain common categories and distinguish display-only pooling from missing values.
    # Input: values=['D1','D1','D2',None], limit=1.
    # Output: ['Value: D1','Value: D1','Other categories (display only)','Missing (not supplied)'].
    # Explanation: D2 is pooled only in the point legend; its original identifier remains in the hover/table.
    # Trick: Prefixing observed strings prevents a real category named Missing from colliding with missing metadata.
    text = values.astype('string')
    keep = text.value_counts().head(limit).index
    labels = ('Value: ' + text).where(text.isin(keep), 'Other categories (display only)')
    return labels.where(text.notna(), 'Missing (not supplied)')


def _point_sizes(values):
    # CORE LOGIC: STEP 1 — Bound marker areas with square-root scaling of known positive sizes.
    # Input: values=[100,400,0,None].
    # Output: marker areas=[76,140,12,12], quantity status=['positive','positive','unknown/nonpositive','unknown/nonpositive'].
    # Explanation: With largest positive size 400, areas are 12 + 128*sqrt(size/400); unknown sizes use the minimum.
    # Trick: Nonpositive and nonfinite quantities remain visible; marker area is a display scale, not an analytical weight.
    numbers = pd.to_numeric(values, errors='coerce').astype('float64')
    positive = numbers.gt(0) & np.isfinite(numbers)
    maximum = numbers.where(positive).max()
    fraction = numbers.where(positive, 0) / maximum if maximum > 0 else pd.Series(0., index=values.index)
    sizes = 12 + 128 * np.sqrt(fraction.fillna(0).clip(0, 1))
    return sizes, pd.Series(np.where(positive, 'positive', 'unknown/nonpositive'), index=values.index)


def _points(rows, config, view_values):
    # CORE LOGIC: STEP 1 — Draw a deterministic uniform sample solely for rendering individual records.
    # Input: source positions=[0,1,2], max_points=3, random_state=0.
    # Output: all positions [0,1,2] are retained in source order; neither errors nor quantities affect selection.
    # Explanation: Larger populations use a seeded sample without replacement and disclose the displayed fraction.
    # Trick: Sorting sampled positions restores source order; bins and inference always use the unsampled population.
    rng = np.random.default_rng(config.random_state)
    positions = np.sort(rng.choice(len(rows), min(len(rows), config.max_points), replace=False))
    points = rows.iloc[positions].copy().reset_index(drop=True)
    # CORE LOGIC: STEP 2 — Color records by supplied side, falling back to dealer or counterparty.
    # Input: side_column='side', dealer_column='dealer', rows=[{'side':'B','dealer':'D1'},
    # {'side':'S','dealer':'D2'}], positions=[1], max_categories=12.
    # Output: __view_side=['Value: S'], __view_color=['Value: S']; original dealer remains 'D2'.
    # Explanation: Side codes determine color without being translated into economic buy/sell conventions.
    # Trick: Pool categories on the full paired population before positional sampling, so a new sample
    # does not change category membership. to_numpy avoids alignment against duplicate source indexes.
    # With no color column, synthetic 'Unmapped' stays distinct from a literal supplied value named 'Unmapped'.
    color_column = config.side_column or config.dealer_column or config.counterparty_column
    side = rows[config.side_column] if config.side_column else pd.Series('Unmapped', index=rows.index)
    side_labels = _labels(side, config.max_categories)
    color = rows[color_column] if color_column else pd.Series('Unmapped', index=rows.index)
    color_labels = side_labels if config.side_column else _labels(color, config.max_categories)
    if color_column is None:
        color_labels = pd.Series('Unmapped', index=rows.index)
    points['__view_side'] = side_labels.iloc[positions].to_numpy()
    points['__view_color'] = color_labels.iloc[positions].to_numpy()
    # CORE LOGIC: STEP 3 — Attach display size and preserve unknown/nonpositive quantity status.
    # Input: plotted quantity=[100,400,0,None], quantity_column='quantity'.
    # Output: __view_size=[76,140,12,12], status=['positive','positive','unknown/nonpositive','unknown/nonpositive'].
    # Explanation: Size controls point area only; every record retains its original analytical weight.
    # Trick: With no quantity column every marker gets the minimum area and unknown/nonpositive status.
    quantity = points[config.quantity_column] if config.quantity_column else pd.Series(np.nan, index=points.index)
    points['__view_size'], points['__view_quantity_status'] = _point_sizes(quantity)
    # CORE LOGIC: STEP 4 — Attach coordinates computed before sampling, without replacing original predictions.
    # Input: positions=[1], full view candidate=[-2,3], actual=[0,0], reference=[1,4], valid=[True,True].
    # Output: sampled __view_candidate=[3], __view_actual=[0], __view_reference=[4], __view_valid=[True].
    # Trick: Positional arrays keep coordinates aligned even if original DataFrame indexes repeat.
    for column in view_values:
        points[column] = view_values[column].iloc[positions].to_numpy()
    return points


def _focused_rows(comparison, config):
    # VALIDATION LOGIC: Entity focus is an exact supplied identifier, never a fuzzy or inferred match.
    entity = comparison.config['entity_column']
    if (config.focus_entity is not None or config.point_view == 'within_entity') and entity is None:
        raise ValueError('Map entity_column before focusing an entity or using within_entity coordinates.')
    if config.focus_entity is None:
        return comparison.rows
    # CORE LOGIC: STEP 1 — Limit only this trade review to the selected entity on the existing paired population.
    # Input: entity IDs=['A','B','A'], focus_entity='A'; paired record IDs=[10,20,30].
    # Output: trade review record IDs=[10,30]; the original comparison still contains [10,20,30].
    # Trick: Equality uses original values and excludes missing IDs; model fitting and other tabs are unaffected.
    keep = comparison.rows['__entity'].eq(config.focus_entity).fillna(False)
    if not keep.any():
        raise ValueError('focus_entity has no paired records in the applied population; select all entities or another ID.')
    return comparison.rows.loc[keep]


def _view_coordinates(rows, config, error_scale):
    # CORE LOGIC: STEP 1 — Start with reconstructed source target levels, preserving all signed errors.
    # Input: actual=[1,2], reference=[1.1,1.8], candidate=[1.05,1.9].
    # Output: view_actual=[1,2], view_reference=[1.1,1.8], view_candidate=[1.05,1.9].
    # Trick: These prediction columns already include configured offsets; never add an anchor again here.
    columns = {'__actual':'__view_actual', '__reference':'__view_reference', '__candidate':'__view_candidate'}
    result = rows[list(columns)].rename(columns=columns).copy()
    # CORE LOGIC: STEP 2 — Expose signed prediction errors directly instead of compressing them into large levels.
    # Input: actual=[1,2], reference=[1.1,1.8], candidate=[1.05,1.9], error_scale=100, point_view='residual'.
    # Output: view_actual=[0,0], view_reference≈[10,-20], view_candidate≈[5,-10] (floating point).
    # Trick: Existing errors are already scaled; multiplying them by error_scale again would be incorrect.
    if config.point_view == 'residual':
        result['__view_actual'] = 0.
        result['__view_reference'] = rows['__error_reference'].to_numpy()
        result['__view_candidate'] = rows['__error_candidate'].to_numpy()
    # CORE LOGIC: STEP 3 — Remove each entity's actual mean using one common center for all three series.
    # Input: entities=['A','A','B',None], actual=[1,1.2,4,5], candidate=[1.05,1.15,4.1,5.1], error_scale=100.
    # Output: actual centers=[1.1,1.1,4,NaN], view_actual≈[-10,10,0,NaN], view_candidate≈[-5,5,10,NaN].
    # Trick: groupby.transform broadcasts full paired entity means before sampling. Missing IDs are not pooled;
    # singleton actuals become zero. The common retrospective center preserves model bias; it is not a predictive feature.
    if config.point_view == 'within_entity':
        center = rows.groupby('__entity', sort=False, observed=True)['__actual'].transform('mean')
        result = result.sub(center.to_numpy(), axis=0) * error_scale
    # CORE LOGIC: STEP 4 — Mark unplottable coordinates explicitly while retaining original rows and metrics.
    # Input: view triples=[(0,2,1),(NaN,NaN,NaN)]. Output: __view_valid=[True,False].
    # Trick: Missing entity centers and numerical overflow only suppress the point display, never the paired error tables.
    result['__view_valid'] = np.isfinite(result.to_numpy()).all(axis=1)
    return result


def trade_tables(comparison, settings=None):
    """Return full paired intraday metrics, sampled points, and an explicit support audit."""
    # CONFIGURATION LOGIC: Caller-selected metadata has no inferred economic meaning.
    config = TradeViewConfig() if settings is None else settings
    config = TradeViewConfig(**config) if isinstance(config, dict) else config
    if not isinstance(config, TradeViewConfig):
        raise TypeError('settings must be TradeViewConfig, a dict or None.')
    for column in (config.side_column, config.counterparty_column, config.dealer_column, config.quantity_column):
        if column is not None and column not in comparison.rows:
            raise ValueError(f'Trade metadata column not found: {column!r}')
    # DIAGNOSTIC LOGIC: Entity focus affects this view only; all other comparison tables retain their applied population.
    rows = _focused_rows(comparison, config)
    times = _time_values(rows)
    frequency, date_count, span = _frequency(times, comparison.config['timezone'], config.frequency)
    bins = _comparison_bins(rows, times, config, frequency, comparison.tolerance, comparison.config['timezone']) if times.notna().any() else pd.DataFrame()
    view_values = _view_coordinates(rows, config, comparison.config['error_scale'])
    points = _points(rows, config, view_values)
    # CORE LOGIC: STEP 1 — Count singleton and unknown identities without inventing a missing-ID entity.
    # Input: entity IDs=['A','A','B',None], actual=[1,2,4,5].
    # Output: entity sizes=[2,2,1,NaN], singleton_entity_rows=1, missing_entity_rows=1.
    # Trick: Singleton entities have no observed within-entity actual variation, even when prediction errors are nonzero.
    entity_sizes = rows.groupby('__entity', sort=False, observed=True)['__actual'].transform('size')
    singleton_rows = int(entity_sizes.eq(1).sum())
    missing_entity_rows = int(rows['__entity'].isna().sum())
    # REPORTING LOGIC: Explain the coordinate transform separately from the unchanged evaluation target.
    notes = {'residual': 'Prediction minus actual, in the configured error unit; zero is an exact prediction.',
             'within_entity': 'Subtract the same full paired actual mean per entity from actual and both models, then apply error_scale. Retrospective diagnostic only, not a predictive feature. Missing IDs are unavailable; singleton actuals are zero.',
             'level': 'Reconstructed target levels in source units. Between-entity level differences can hide economically important errors; use residuals or focus one entity.'}
    # REPORTING LOGIC: Keep source-level units separate from scaled residual and centered diagnostic coordinates.
    summary = dict(paired_rows=len(rows), input_paired_rows=len(comparison.rows), timed_rows=int(times.notna().sum()),
                   missing_time_rows=int(times.isna().sum()), local_dates=date_count, calendar_span_days=span,
                   frequency=frequency, timezone=comparison.config['timezone'], clock='UTC fixed elapsed bins; local labels',
                   plotted_rows=int(points['__view_valid'].sum()), sampled_rows=len(points),
                   sampled=len(points) < len(rows), random_state=config.random_state,
                   total_bins=len(bins), supported_bins=int(bins['supported'].sum()) if len(bins) else 0,
                   empty_bins=int(bins['n'].eq(0).sum()) if len(bins) else 0, min_count=config.min_count,
                   id_column=comparison.config['id_column'], entity_column=comparison.config['entity_column'],
                   level_unit='source target units', error_unit=comparison.unit,
                   actual_column=comparison.config['actual'], point_view=config.point_view, focus_entity=config.focus_entity,
                   view_note=notes[config.point_view], missing_view_rows=int((~view_values['__view_valid']).sum()),
                   missing_entity_rows=missing_entity_rows, singleton_entity_rows=singleton_rows,
                   color_column=config.side_column or config.dealer_column or config.counterparty_column,
                   color_role='side' if config.side_column else 'dealer' if config.dealer_column else
                              'counterparty' if config.counterparty_column else 'unmapped',
                   side_encoding='color', symbol_encoding='prediction error only' if config.point_view == 'residual' else 'actual versus prediction',
                   point_sampling='seeded uniform without replacement; full population metrics',
                   size_scaling='marker area = 12 + 128*sqrt(quantity / largest positive sampled quantity); unknown = 12')
    return dict(summary=pd.DataFrame([summary]), intraday=bins, points=points, settings=asdict(config))
