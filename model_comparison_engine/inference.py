"""Paired loss tests on independent records or equally weighted date/entity means.

The two-sided one-sample t test targets the mean candidate-minus-reference loss
difference. Grouped units change that estimand to an equal-unit mean. They do
not provide row-weighted cluster-robust inference. Chosen units must be
independent, with an approximately normal mean; small samples additionally
need approximately normal unit differences. Date aggregation does not remove
serial dependence. Per-table multiplicity correction does not account for
repeated exploratory filtering, model selection, or repeated report viewing.
Confidence intervals are pointwise, not simultaneous or multiplicity adjusted.
"""
# SETUP LOGIC: Statistical inference is independent of plotting and model fitting.
from dataclasses import dataclass
from numbers import Integral, Real
import numpy as np
import pandas as pd
from scipy import stats


# CONFIGURATION LOGIC: Stable output names also define empty inference tables.
RESULT_COLUMNS = [
    'row_count', 'n', 'paired_n', 'tested_n', 'excluded_unit_rows', 'unit_count',
    'mean_loss_difference', 't_statistic', 'df', 'p_value', 'q_value',
    'ci_low', 'ci_high', 'significant', 'significance', 'low_support', 'status',
    'loss', 'inference_unit', 'correction', 'alpha', 'min_units',
]
INFERENCE_ASSUMPTIONS = (
    'Chosen inference units must be independent, with an approximately normal mean; '
    'small samples additionally require approximately normal unit differences. '
    'Date/entity means receive equal weight and are not row-weighted cluster-robust tests. '
    'Date aggregation does not remove serial dependence. Correction covers valid tests '
    'in this displayed table only, not repeated exploration or model selection. '
    'Confidence intervals are pointwise and unadjusted.'
)


@dataclass(frozen=True)
class InferenceConfig:
    """Declare loss, inference unit, and a per-table false-discovery-rate rule."""
    # CONFIGURATION LOGIC: The comparison API resolves auto from its time-column configuration.
    loss: str = 'absolute'
    unit: str = 'auto'
    correction: str = 'by'
    alpha: float = .05
    min_units: int = 10

    def __post_init__(self):
        # VALIDATION LOGIC: Reject unknown modes and invalid thresholds before reading data.
        if self.loss not in {'absolute', 'squared'}:
            raise ValueError('Inference loss must be absolute or squared.')
        if self.unit not in {'auto', 'record', 'date', 'entity'}:
            raise ValueError('Inference unit must be auto, record, date or entity.')
        if self.correction not in {'by', 'bh', 'none'}:
            raise ValueError('Inference correction must be by, bh or none.')
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real) or not 0 < self.alpha < 1:
            raise ValueError('Inference alpha must be finite and strictly between zero and one.')
        if isinstance(self.min_units, bool) or not isinstance(self.min_units, Integral) or self.min_units < 2:
            raise ValueError('Inference min_units must be an integer of at least two.')


def _loss_differences(rows, loss):
    # CORE LOGIC: STEP 1 — Derive both losses from each record before pairing their difference.
    # Input: reference errors=[-2,3], candidate errors=[1,-1], loss='squared'.
    # Output: reference losses=[4,9], candidate losses=[1,1], differences=[-3,-8].
    # Explanation: The squared losses change by 1−4=-3 and 1−9=-8 on their original records.
    # Trick: Squaring uses float64; overflow stays nonfinite and makes the cell untested.
    prefix = '__ae_' if loss == 'absolute' else '__error_'
    reference = rows[prefix+'reference'].to_numpy(dtype=float, na_value=np.nan)
    candidate = rows[prefix+'candidate'].to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(over='ignore', invalid='ignore'):
        if loss == 'squared':
            reference, candidate = np.square(reference), np.square(candidate)
        differences = candidate-reference
    # CORE LOGIC: STEP 2 — Keep an explicit validity mask without concealing invalid paired losses.
    # Input: reference losses=[4,inf], candidate losses=[1,1], differences=[-3,-inf].
    # Output: valid=[True,False], differences=[-3,-inf].
    # Explanation: The second pair contains an infinite source loss, so its validity flag is false.
    # Trick: Finite differences alone cannot certify the two source losses were valid.
    valid = np.isfinite(reference) & np.isfinite(candidate) & np.isfinite(differences)
    return differences, valid


def _unit_differences(rows, differences, valid, unit):
    # CORE LOGIC: STEP 1 — Restrict inference to complete pairs with known chosen units.
    # Input: differences=[-2,-4,6], valid=[True,True,True], dates=['D1',None,'D2'], unit='date'.
    # Output: known=[True,False,True], eligible=[True,False,True], tested_n=2, excluded_unit_rows=1.
    # Explanation: The middle record has paired predictions but no date to define an inference unit.
    # Trick: Position-based masks preserve exact pairing even when the DataFrame index repeats.
    known = np.ones(len(rows), dtype=bool) if unit == 'record' else rows['__'+unit].notna().to_numpy()
    eligible = valid & known
    tested_n = int(eligible.sum())
    excluded_unit_rows = int((valid & ~known).sum())
    # CORE LOGIC: STEP 2 — Normalize finite differences before taking equally weighted unit means.
    # Input: differences=[-2,-4,6], eligible=[True,True,True], dates=['D1','D1','D2'], unit='date'.
    # Output: scale=6, normalized=[-1/3,-2/3,1], observations=[-.5,1] (approximately).
    # Explanation: D1 averages its two normalized differences to -.5; D2 contributes its single value 1.
    # Trick: One common scale avoids overflowing group sums; repeated dates contribute one mean each.
    selected = differences[eligible]
    scale = float(np.max(np.abs(selected))) if tested_n else 0.0
    normalized = selected/scale if scale else selected
    if unit == 'record':
        observations = normalized
    else:
        units = rows.loc[eligible, '__'+unit].to_numpy()
        grouped = pd.DataFrame({'unit': units, 'difference': normalized})
        observations = grouped.groupby('unit', observed=True, sort=False)['difference'].mean().to_numpy()
    # CORE LOGIC: STEP 3 — Rescale unit means after any within-unit cancellation.
    # Input: observations=[-.125,.25], scale=8 (date means originally [-1,2]).
    # Output: unit_scale=.25, observations=[-.5,1], scale=2; original date means remain [-1,2].
    # Explanation: Dividing each observation by .25 and multiplying the scale by .25 preserves each loss difference.
    # Trick: A second common scale prevents squaring tiny normalized unit means from underflowing in the t test.
    unit_scale = float(np.max(np.abs(observations))) if len(observations) else 0.0
    if unit_scale:
        observations = observations/unit_scale
        scale *= unit_scale
    return observations, scale, tested_n, excluded_unit_rows


def _blank_result(rows, valid, observations, scale, tested_n, excluded, config, min_count):
    # CORE LOGIC: STEP 1 — Preserve support and the chosen-unit effect even when no test is available.
    # Input: row_count=3, valid=[T,T,T], observations=[-.5,1], scale=6, tested_n=3, excluded=0,
    #        config=InferenceConfig(unit='date',min_units=2), min_count=4.
    # Output: paired_n=n=3, unit_count=2, mean_loss_difference=1.5, low_support=True.
    # Explanation: The two date means are -3 and 6, whose equal-weight mean is 1.5; three rows fail the four-row minimum.
    # Trick: Multiplying after averaging keeps a representable mean finite for large losses.
    paired_n, unit_count = int(valid.sum()), len(observations)
    mean = float(np.mean(observations)*scale) if unit_count else np.nan
    low_support = tested_n < min_count or unit_count < config.min_units
    result = dict(row_count=len(rows), n=paired_n, paired_n=paired_n, tested_n=tested_n,
                  excluded_unit_rows=excluded, unit_count=unit_count, mean_loss_difference=mean)
    # REPORTING LOGIC: Every untested cell starts with empty inferential quantities and no stars.
    result.update({name: np.nan for name in ['t_statistic', 'df', 'p_value', 'q_value', 'ci_low', 'ci_high']})
    result.update(significant=False, significance='', low_support=bool(low_support), status='untested')
    result.update(loss=config.loss, inference_unit=config.unit, correction=config.correction,
                  alpha=config.alpha, min_units=config.min_units)
    return result


def _test_cell(rows, config, min_count):
    # CORE LOGIC: STEP 1 — Construct the paired observations and their explicit support audit.
    # Input: reference abs=[2,4,6], candidate abs=[1,2,3], unit='record', min_units=2, min_count=3.
    # Output: differences=[-1,-2,-3], observations≈[-1/3,-2/3,-1], scale=3, unit_count=3, mean=-2.
    # Explanation: Each record remains a separate unit and the three paired differences average to -2.
    # Trick: The same pairing routine serves global, one-way, and interaction tables.
    differences, valid = _loss_differences(rows, config.loss)
    observations, scale, tested_n, excluded = _unit_differences(rows, differences, valid, config.unit)
    result = _blank_result(rows, valid, observations, scale, tested_n, excluded, config, min_count)
    # CORE LOGIC: STEP 2 — Reject invalid loss cells or inadequate row/unit support before testing.
    # Input: valid=[T,T,T], tested_n=3, unit_count=3, min_count=30, min_units=2.
    # Output: status='low_rows'; t_statistic,p_value,q_value,ci_low,ci_high remain NaN; significance=''.
    # Explanation: Three eligible records fall below the requested thirty-record support threshold.
    # Trick: A single overflowed loss invalidates the cell rather than silently changing its population.
    if not valid.all():
        result['status'] = 'invalid_loss'
    elif tested_n < min_count:
        result['status'] = 'low_rows'
    elif len(observations) < config.min_units:
        result['status'] = 'low_units'
    if result['status'] != 'untested':
        return result
    # CORE LOGIC: STEP 3 — Detect constant and numerically indistinguishable unit differences.
    # Input: observations=[1,1,1], scale=2, unit_count=3.
    # Output: deviation=0, status='zero_variance'; no t statistic, confidence interval, or stars.
    # Explanation: Identical unit differences have no estimable nonzero standard error for a t test.
    # Trick: The relative threshold uses normalized magnitudes, so a small measurement unit alone is safe.
    deviation = float(np.std(observations, ddof=1))
    if deviation == 0:
        result['status'] = 'zero_variance'
    elif deviation <= 16*np.finfo(float).eps*max(float(np.max(np.abs(observations))), np.finfo(float).tiny):
        result['status'] = 'near_zero_variance'
    if result['status'] != 'untested':
        return result
    # CORE LOGIC: STEP 4 — Apply a two-sided one-sample t test and a pointwise interval to unit differences.
    # Input: observations=[-1/3,-2/3,-1], scale=3, alpha=.05.
    # Output: t≈-3.464102, df=2, p≈.0741799, mean=-2, ci≈[-4.484138,.484138].
    # Explanation: The unscaled differences [-1,-2,-3] have mean -2 and sample standard deviation 1.
    # Trick: A shared positive scale leaves t/p unchanged; only the effect and its interval regain loss units.
    test = stats.ttest_1samp(observations, popmean=0, alternative='two-sided')
    degrees = len(observations)-1
    radius = stats.t.ppf(1-config.alpha/2, degrees)*deviation/np.sqrt(len(observations))
    with np.errstate(over='ignore', invalid='ignore'):
        lower = (float(np.mean(observations))-radius)*scale
        upper = (float(np.mean(observations))+radius)*scale
    values = [float(test.statistic), degrees, float(test.pvalue), lower, upper]
    # CORE LOGIC: STEP 5 — Publish inferential results only when every computed quantity is finite.
    # Input: values=[-3.464102,2,.0741799,-4.484138,.484138] (rounded).
    # Output: status='ok', t_statistic≈-3.464102, df=2, p_value≈.0741799, ci_low≈-4.484138, ci_high≈.484138.
    # Explanation: All five test outputs and the effect are finite, so this cell is eligible for multiplicity correction.
    # Trick: Even a finite t statistic is withheld if restoring loss units overflows either interval endpoint.
    if np.isfinite(values).all() and np.isfinite(result['mean_loss_difference']):
        result.update(zip(['t_statistic', 'df', 'p_value', 'ci_low', 'ci_high'], values))
        result['status'] = 'ok'
    else:
        result['status'] = 'invalid_result'
    return result


def _adjust_pvalues(values, correction):
    # CORE LOGIC: STEP 1 — Rank valid p values and compute the declared false-discovery-rate bound.
    # Input: values=[.04,.001,.03], correction='bh'.
    # Output: order=[1,2,0], ordered=[.001,.03,.04], factor=1, adjusted=[.003,.045,.04].
    # Explanation: BH multiplies each ordered p value by three divided by its one-based rank.
    # Trick: BY multiplies by the harmonic sum for arbitrary dependence; BH needs independence or suitable positive dependence.
    values = np.asarray(values, dtype=float)
    if correction == 'none' or not len(values):
        return values.copy()
    order = np.argsort(values, kind='stable')
    ranks = np.arange(1, len(values)+1, dtype=float)
    factor = np.sum(1/ranks) if correction == 'by' else 1.0
    adjusted = values[order]*(len(values)/ranks)*factor
    # CORE LOGIC: STEP 2 — Enforce monotonic adjusted values, cap at one, then restore original cell order.
    # Input: order=[1,2,0], adjusted=[.003,.045,.04].
    # Output: monotone=[.003,.04,.04], restored q=[.04,.003,.04].
    # Explanation: The middle ranked bound falls from .045 to .04 before values return to their original cells.
    # Trick: A reverse cumulative minimum implements the step-up correction, including tied p values.
    monotone = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(values)
    restored[order] = np.clip(monotone, 0, 1)
    return restored


def grouped_tests(rows, keys, config, min_count=30):
    """Test displayed groups, correcting only valid tests within this returned table.

    ``n`` and ``paired_n`` count finite paired losses. ``tested_n`` counts paired
    records with known inference-unit labels, even when support/variance rules
    subsequently prevent a test. ``unit_count`` counts their independent-unit
    observations. ``excluded_unit_rows`` counts paired records with missing unit
    labels. Negative effects and t statistics favor the candidate. Stars use
    q <= .001/.01/.05, separately from ``significant`` at the configured alpha.
    """
    # VALIDATION LOGIC: Unit selection must be resolved by the public comparison API.
    if not isinstance(config, InferenceConfig):
        raise TypeError('Pass an InferenceConfig instance.')
    if config.unit == 'auto':
        raise ValueError('Resolve inference unit=auto from the comparison time-column configuration first.')
    if isinstance(min_count, bool) or not isinstance(min_count, Integral) or min_count < 1:
        raise ValueError('min_count must be a positive integer.')
    keys = list(keys)
    if len(set(keys)) != len(keys) or set(keys) & set(RESULT_COLUMNS):
        raise ValueError('Grouping keys must be unique and must not use inference result column names.')
    # CORE LOGIC: STEP 1 — Test each observed group, or a single global cell when keys are empty.
    # Input: group=['A','A','B'], reference abs=[2,4,3], candidate abs=[1,2,1], keys=['group'], min_count=2.
    # Output: two cells in order A,B; A has row_count=2, mean=-1.5; B has row_count=1, mean=-2.
    # Explanation: A pairs differences [-1,-2] and B pairs [-2], with no invented empty group.
    # Trick: observed=True avoids fabricated categorical combinations; dropna=False preserves missing groups.
    grouped = rows.groupby(keys[0] if len(keys) == 1 else keys, observed=True, sort=True, dropna=False) if keys else [((), rows)]
    records = []
    for labels, subset in grouped:
        labels = (labels,) if len(keys) == 1 else labels
        records.append(dict(zip(keys, labels), **_test_cell(subset, config, min_count)))
    table = pd.DataFrame(records, columns=[*keys, *RESULT_COLUMNS])
    # CORE LOGIC: STEP 2 — Adjust only finite, eligible displayed tests and retain untested cells without stars.
    # Input: p_value=[.04,.001,NaN,.03], status=['ok','ok','low_rows','ok'], correction='bh', alpha=.05.
    # Output: q_value=[.04,.003,NaN,.04], significant=[T,T,F,T], significance=['*','**','','*'].
    # Explanation: The low-support cell contributes no hypothesis to the three-test correction family.
    # Trick: The correction family is this table; pointwise confidence intervals are never presented as adjusted intervals.
    available = table['status'].eq('ok') & np.isfinite(table['p_value'].to_numpy(dtype=float))
    table.loc[available, 'q_value'] = _adjust_pvalues(table.loc[available, 'p_value'], config.correction)
    table.loc[available, 'significant'] = table.loc[available, 'q_value'].le(config.alpha)
    qvalues = table.loc[available, 'q_value'].to_numpy(dtype=float)
    table.loc[available, 'significance'] = np.select([qvalues <= .001, qvalues <= .01, qvalues <= .05], ['***', '**', '*'], default='')
    # REPORTING LOGIC: Metadata records the interpretation without duplicating caveats in each cell.
    table.attrs.update(assumptions=INFERENCE_ASSUMPTIONS, family='valid displayed cells in this table',
                       estimand='record mean' if config.unit == 'record' else 'equal-'+config.unit+' mean',
                       confidence_interval='pointwise, unadjusted', alternative='two-sided')
    return table
