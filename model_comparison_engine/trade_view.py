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

    def __post_init__(self):
        # VALIDATION LOGIC: Bound displays and reject non-fixed clock intervals before allocation.
        if self.frequency != 'auto':
            _bin_nanoseconds(self.frequency)
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


def _comparison_bins(comparison, times, config, frequency):
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
    rows = comparison.rows.loc[known].assign(bin=positions)
    # CORE LOGIC: STEP 2 — Recompute paired metrics using all records before adding explicit empty bins.
    # Input: bin=[0,0,2], reference errors=[-1,3,2], candidate errors=[0,2,1], min_count=2.
    # Output: n=[2,0,1], reference_mae=[2,NaN,2], candidate_mae=[1,NaN,1], supported=[True,False,False].
    # Explanation: Plot sampling never changes these metrics; a missing bin cannot appear as a zero error.
    # Trick: Reindex after aggregation adds time gaps while preserving undefined losses and zero record counts.
    result = grouped_metrics(rows, ['bin'], tolerance=comparison.tolerance, min_count=config.min_count)
    result = result.set_index('bin').reindex(range(len(grid)))
    result[['n', 'entities', 'dates']] = result[['n', 'entities', 'dates']].fillna(0).astype('int64')
    result['supported'] = result['n'].ge(config.min_count)
    result['low_support'] = ~result['supported']
    result.insert(0, 'time', grid.tz_convert(comparison.config['timezone']))
    result.insert(1, 'time_end', (grid + pd.Timedelta(bin_ns, unit='ns')).tz_convert(comparison.config['timezone']))
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


def _points(comparison, config):
    # CORE LOGIC: STEP 1 — Draw a deterministic uniform sample solely for rendering individual records.
    # Input: source positions=[0,1,2], max_points=3, random_state=0.
    # Output: all positions [0,1,2] are retained in source order; neither errors nor quantities affect selection.
    # Explanation: Larger populations use a seeded sample without replacement and disclose the displayed fraction.
    # Trick: Sorting sampled positions restores source order; bins and inference always use the unsampled population.
    rows = comparison.rows
    rng = np.random.default_rng(config.random_state)
    positions = np.sort(rng.choice(len(rows), min(len(rows), config.max_points), replace=False))
    points = rows.iloc[positions].copy().reset_index(drop=True)
    # CORE LOGIC: STEP 2 — Attach explicit role metadata without interpreting the caller's category codes.
    # Input: side_column='side', dealer_column='dealer', rows=[{'side':'B','dealer':'D1'}].
    # Output: __view_side=['Value: B'], __view_color=['Value: D1']; B is never renamed Buy or Sell.
    # Explanation: Dealer supplies color when mapped, otherwise counterparty does; both original fields remain available.
    # Trick: Category limits bound legends and marker symbols, not evaluation populations or model features.
    side = points[config.side_column] if config.side_column else pd.Series('Unmapped', index=points.index)
    color_column = config.dealer_column or config.counterparty_column
    color = points[color_column] if color_column else pd.Series('Unmapped', index=points.index)
    points['__view_side'] = _labels(side, 5)
    points['__view_color'] = _labels(color, config.max_categories)
    quantity = points[config.quantity_column] if config.quantity_column else pd.Series(np.nan, index=points.index)
    points['__view_size'], points['__view_quantity_status'] = _point_sizes(quantity)
    return points


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
    # DIAGNOSTIC LOGIC: Prepared paired rows preserve offset reconstruction and the configured error scale.
    times = _time_values(comparison.rows)
    frequency, date_count, span = _frequency(times, comparison.config['timezone'], config.frequency)
    bins = _comparison_bins(comparison, times, config, frequency) if times.notna().any() else pd.DataFrame()
    points = _points(comparison, config)
    # REPORTING LOGIC: The point view is source-unit level data; binned losses use the declared error display unit.
    summary = dict(paired_rows=len(comparison.rows), timed_rows=int(times.notna().sum()),
                   missing_time_rows=int(times.isna().sum()), local_dates=date_count, calendar_span_days=span,
                   frequency=frequency, timezone=comparison.config['timezone'], clock='UTC fixed elapsed bins; local labels',
                   plotted_rows=len(points), sampled=len(points) < len(comparison.rows), random_state=config.random_state,
                   total_bins=len(bins), supported_bins=int(bins['supported'].sum()) if len(bins) else 0,
                   empty_bins=int(bins['n'].eq(0).sum()) if len(bins) else 0, min_count=config.min_count,
                   id_column=comparison.config['id_column'], entity_column=comparison.config['entity_column'],
                   level_unit='source target units', error_unit=comparison.unit,
                   point_sampling='seeded uniform without replacement; full population metrics',
                   size_scaling='marker area = 12 + 128*sqrt(quantity / largest positive plotted quantity); unknown = 12')
    return dict(summary=pd.DataFrame([summary]), intraday=bins, points=points, settings=asdict(config))
