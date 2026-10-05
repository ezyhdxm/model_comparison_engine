"""Reusable categorical, fixed-bin and two-way slices, with explicit missing groups."""
# SETUP LOGIC: Slice specifications contain metadata only.
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Slice:
    # CONFIGURATION LOGIC: right=True means (a,b]; right=False means [a,b).
    column: str
    bins: object = None
    labels: object = None
    right: bool = True
    top_n: int = 20
    name: str = None

    def to_dict(self):
        # SERIALIZATION LOGIC: Preserve the declared boundaries alongside exported results.
        return asdict(self)


def _unused(existing, name):
    # FORMATTING LOGIC: Synthetic categories must not collide with literal input values.
    while name in existing:
        name += '*'
    return name


def slice_labels(frame, spec):
    # VALIDATION LOGIC: Reject incomplete or misleading bins instead of silently dropping rows.
    if spec.column not in frame:
        raise ValueError(f'Unknown slice column: {spec.column!r}')
    if spec.top_n < 1:
        raise ValueError('top_n must be at least 1.')
    source = frame[spec.column]
    if spec.bins is not None:
        edges = np.asarray(spec.bins, dtype=float)
        if len(edges) < 2 or np.isnan(edges).any() or not (np.diff(edges) > 0).all():
            raise ValueError('Bin edges must be strictly increasing; +/- infinity is allowed.')
        if spec.labels is not None and len(set(spec.labels)) != len(spec.labels):
            raise ValueError('Bin labels must be unique.')
        # CORE LOGIC: STEP 1 — Assign fixed intervals, retaining missing and out-of-range values separately.
        # Input: x=[0,1,2,3,NaN], bins=[0,1,2], right=True, labels=['low','high'].
        # Output: ['low','low','high','(outside bins)','(missing/nonfinite)'].
        # Explanation: include_lowest puts 0 in the first interval; 1 is its closed upper boundary.
        # Trick: Missing and finite-but-uncovered values cannot be silently merged or dropped.
        values = pd.to_numeric(source, errors='coerce').replace([np.inf, -np.inf], np.nan)
        grouped = pd.cut(values, edges, labels=spec.labels, right=spec.right, include_lowest=True)
        order = [str(v) for v in grouped.cat.categories]
        labels = grouped.astype('string')
        outside, missing = _unused(order, '(outside bins)'), _unused(order, '(missing/nonfinite)')
        labels = labels.mask(values.notna() & labels.isna(), outside).fillna(missing)
        return pd.Series(pd.Categorical(labels, categories=order+[outside, missing], ordered=True), index=source.index)
    # CORE LOGIC: STEP 2 — Bound categorical displays by frequency, while preserving a distinct missing group.
    # Input: x=['A','A','B','C',None], top_n=1.
    # Output: ['A','A','(other)','(other)','(missing)'].
    # Explanation: A has the greatest support; B and C are combined and the missing row stays visible.
    # Trick: Categories are selected by counts, never by model gains; ties use stable lexical order.
    labels = source.astype('string')
    counts = labels.dropna().value_counts().rename_axis('label').reset_index(name='n')
    keep = counts.sort_values(['n','label'], ascending=[False,True], kind='stable').head(spec.top_n)['label'].tolist()
    other, missing = _unused(set(labels.dropna()), '(other)'), _unused(set(labels.dropna()), '(missing)')
    labels = labels.where(labels.isin(keep) | labels.isna(), other).fillna(missing)
    return pd.Series(pd.Categorical(labels, categories=keep+[other, missing], ordered=True), index=source.index)


def default_slices(columns):
    """Return generic derived time slices; other metadata needs an explicit Slice."""
    # CONFIGURATION LOGIC: Keep the columns argument for API stability; no field names imply domain bins.
    return [Slice('__hour', [-.5,5.5,11.5,17.5,23.5],
                  ['00–05','06–11','12–17','18–23'], name='Hour'),
            Slice('__date', top_n=40, name='Date')]
