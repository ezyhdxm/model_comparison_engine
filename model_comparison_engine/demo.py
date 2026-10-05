"""Deterministic synthetic records for learning the interface, never performance evidence."""
# SETUP LOGIC: Generation needs no private files, external services or fitted models.
import numpy as np
import pandas as pd


def make_demo(n=1200, seed=2026):
    # VALIDATION LOGIC: A record count must be an integer; empty synthetic tables remain supported.
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('n must be a nonnegative integer.')
    # CORE LOGIC: STEP 1 — Generate one reproducible population and its reference errors.
    # Input: n=2, seed=0.
    # Output: row_id=[0,1], actual≈[0.12573022,-0.13210486], reference_error≈[0.32021133,0.05245006].
    # Explanation: The numerical examples are rounded to eight decimal places; records retain full precision.
    # Trick: A local generator leaves the caller's global random state unchanged.
    rng = np.random.default_rng(seed)
    row_id = np.arange(n)
    actual = rng.normal(0,1,n)
    reference_error = rng.normal(0,.5,n)
    # CORE LOGIC: STEP 2 — Add a reproducible, heterogeneous candidate error to the same actual values.
    # Input: row_id=[0,1], actual≈[0.12573022,-0.13210486], reference_error≈[0.32021133,0.05245006], next noise≈[-0.05356694,0.03615951].
    # Output: segment=['A','B'], category=['X','Y']; reference≈[0.44594155,-0.07965480], candidate≈[0.37833234,-0.05398531].
    # Explanation: Segment A has an extra synthetic offset of 0.05; these choices demonstrate slicing only.
    # Trick: Both predictions share row identities and actuals, while correlated errors make paired comparisons useful.
    segment = np.take(['A','B','C'],row_id % 3)
    category = np.take(['X','Y'],row_id % 2)
    candidate_error = .8*reference_error+rng.normal(0,.1,n)+np.where(segment=='A',.05,0)
    reference_prediction = actual+reference_error
    candidate_prediction = actual+candidate_error
    # CORE LOGIC: STEP 3 — Attach neutral metadata without changing any prediction or its row alignment.
    # Input: row_id=[0,1], actual≈[.12573022,-.13210486], reference≈[.44594155,-.07965480], candidate≈[.37833234,-.05398531], segment=['A','B'], category=['X','Y'].
    # Output: rows≈[(0,'entity_00',.12573022,.44594155,.37833234,'A','X',.5,1,'2026-01-01T00:00:00Z'), (1,'entity_01',-.13210486,-.07965480,-.05398531,'B','Y',1.5,2,'2026-01-01T00:15:00Z')].
    # Explanation: The returned columns are row_id, entity_id, actual, both prediction columns, segment, category, both measures and time.
    # Trick: Metadata cycles are independent of evaluated errors; all timestamps are explicitly UTC.
    return pd.DataFrame(dict(row_id=row_id,entity_id=[f'entity_{i % 12:02d}' for i in row_id],actual=actual,
                             reference_prediction=reference_prediction,candidate_prediction=candidate_prediction,
                             segment=segment,category=category,measure_1=(row_id % 10)+.5,measure_2=(row_id % 5)+1,
                             time=pd.date_range('2026-01-01',periods=n,freq='15min',tz='UTC')))
