"""Bounded, inference-only explanations of fitted scalar regression models.

Permutation measures held-out predictive reliance, not causal importance. Tree SHAP
explains predictions, not their accuracy. Caller-supplied rows define the population.
"""
# SETUP LOGIC: Optional tree libraries are imported only when their explanation is requested.
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd
from .data import read_data
from .engine import Model, _predict


@dataclass
class ExplanationResult:
    # REPORTING LOGIC: Tables and settings are portable; fitted models are never serialized here.
    tables: dict
    settings: dict
    notes: list = field(default_factory=list)


def _positive_integer(value, name, upper):
    # VALIDATION LOGIC: Hard bounds prevent accidental full-population, thousands-of-predictions jobs.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= upper:
        raise ValueError(f'{name} must be an integer between 1 and {upper:,}.')
    return int(value)


def _validate_model(model, error_scale):
    # VALIDATION LOGIC: Only explicit fitted scalar regressors and a positive output conversion are supported.
    if not isinstance(model, Model):
        raise TypeError('Use Model(name, fitted_estimator, ordered_features, offset=None).')
    if not model.features or len(set(model.features)) != len(model.features):
        raise ValueError('The model needs a nonempty, unique, ordered feature list.')
    if getattr(model.estimator, '_estimator_type', None) == 'classifier':
        raise ValueError('Explanations require scalar regression predictions, not a classifier.')
    if not np.isfinite(error_scale) or error_scale <= 0:
        raise ValueError('error_scale must be positive and finite.')


def _notify(progress, stage, completed, total):
    # UI LOGIC: A caller may update a progress bar or raise an exception to cancel between batches.
    if progress is not None:
        progress(dict(stage=stage, completed=completed, total=total))


def _number(frame, column):
    # CORE LOGIC: STEP 1 — Parse numeric metadata without modifying model feature dtypes.
    # Input: column values=['2.5', 'bad', None, float('inf')].
    # Output: array([2.5, NaN, NaN, NaN]).
    # Explanation: Invalid targets or offsets cannot be used for numerical error evaluation.
    # Trick: Feature columns are never passed through this conversion; categorical model inputs stay categorical.
    values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    return np.where(np.isfinite(values), values, np.nan)


def _input_sample(data, model, actual, max_rows, random_state, row_positions=None):
    # VALIDATION LOGIC: Preserve caller row positions independently of possibly duplicated pandas index labels.
    frame = data if isinstance(data, pd.DataFrame) else read_data(data)
    required = list(model.features) + ([actual] if actual else []) + ([model.offset] if model.offset else [])
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f'Missing explanation columns: {sorted(missing)}')
    # CORE LOGIC: STEP 1 — Identify rows with usable fixed offsets and, when requested, actual values.
    # Input: actual=[101,NaN,103], offset=[100,100,NaN].
    # Output: eligible source positions=[0], actual=[101,NaN,103], offset=[100,100,NaN].
    # Explanation: An unavailable reconstruction offset is not replaced by zero; an omitted offset is zero.
    # Trick: Masks use positional NumPy arrays, so duplicate dataframe index labels do not duplicate records.
    offset = _number(frame, model.offset) if model.offset else np.zeros(len(frame))
    target = _number(frame, actual) if actual else np.full(len(frame), np.nan)
    valid = np.isfinite(offset) & (np.isfinite(target) if actual else True)
    eligible = np.flatnonzero(valid)
    # CONFIGURATION LOGIC: Explicit positions are useful for a reviewed worst case; they bypass random sampling.
    if row_positions is not None:
        selected = np.asarray(list(row_positions))
        if selected.ndim != 1 or selected.dtype.kind not in 'iu' or len(set(selected)) != len(selected):
            raise ValueError('row_positions must be distinct integer positions in the supplied data.')
        if len(selected) > max_rows or (selected < 0).any() or (selected >= len(frame)).any():
            raise ValueError('row_positions exceed max_rows or the supplied data bounds.')
        if not valid[selected].all():
            raise ValueError('A requested row has an invalid actual value or reconstruction offset.')
    else:
        # CORE LOGIC: STEP 2 — Use a reproducible bounded sample in original source order.
        # Input: eligible positions=[0,1,2,3,4], max_rows=3, random_state=0.
        # Output: selected positions=[2,3,4].
        # Explanation: A seeded sample without replacement is taken before any prediction call.
        # Trick: Sorting sampled positions restores source order; sampling is independent of error or feature values.
        rng = np.random.default_rng(random_state)
        selected = np.sort(rng.choice(eligible, max_rows, replace=False)) if len(eligible) > max_rows else eligible
    # VALIDATION LOGIC: Empty explanations are explicit errors, not zero importance.
    if not len(selected):
        raise ValueError('No eligible records remain for explanation.')
    return frame.iloc[selected].copy(), selected, target[selected], offset[selected], len(frame), len(eligible)


def _regression_prediction(frame, model):
    # INFERENCE LOGIC: Preserve the estimator's own prediction defaults, including a selected best iteration.
    try:
        return np.asarray(_predict(frame, model), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{model.name}: prediction failed; supply the same prepared feature dtypes used at training. {exc}') from exc


def _check_saved_prediction(frame, raw, column):
    # VALIDATION LOGIC: A mutated/refitted estimator must not explain predictions saved by an earlier model state.
    if column is None:
        return
    if column not in frame:
        raise ValueError(f'Missing saved prediction column: {column!r}')
    expected = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    if not np.allclose(raw, expected, equal_nan=True, rtol=1e-7, atol=1e-8):
        raise ValueError('Fresh model predictions do not match the saved comparison predictions. Rebuild the comparison with the current fitted model before explaining it.')


def _errors(actual, raw, offset, error_scale):
    # CORE LOGIC: STEP 1 — Reconstruct the final target before measuring errors in the displayed unit.
    # Input: actual=[101,104], raw=[2,1], offset=[100,100], error_scale=1.
    # Output: prediction=[102,101], error=[1,-3].
    # Explanation: A residual-target model is evaluated after adding each record's fixed offset.
    # Trick: Offset values belong to the original rows and are never permuted with feature inputs.
    with np.errstate(over='ignore', invalid='ignore'):
        prediction = raw + offset
        error = (prediction - actual) * error_scale
    return prediction, error


def _losses(error):
    # CORE LOGIC: STEP 1 — Calculate MAE and RMSE using scaling to avoid squaring overflow.
    # Input: error=[3,-4].
    # Output: mae=3.5, rmse=3.5355339059327378 (approximately).
    # Explanation: RMSE is sqrt((9+16)/2); the normalization is algebraically equivalent.
    # Trick: A maximum absolute error of zero uses scale=1 and still produces exact zero losses.
    scale = float(np.max(np.abs(error))) or 1.0
    normalized = error / scale
    return dict(mae=float(np.mean(np.abs(normalized)) * scale),
                rmse=float(np.sqrt(np.mean(normalized ** 2)) * scale))


def _feature_groups(model, groups, include_individual):
    # VALIDATION LOGIC: Labels are unique; requested group members must be actual model inputs.
    if groups is not None and not isinstance(groups, dict):
        raise TypeError('groups must map a readable group name to a nonempty list of model feature columns.')
    result = {str(column): [column] for column in model.features} if include_individual else {}
    for name, columns in (groups or {}).items():
        if not isinstance(name, str) or not name or name in result:
            raise ValueError('Group names must be nonempty strings distinct from individual feature labels.')
        if isinstance(columns, str) or not columns or len(set(columns)) != len(columns):
            raise ValueError(f'Group {name!r} must contain a nonempty list of distinct features.')
        if set(columns) - set(model.features):
            raise ValueError(f'Group {name!r} includes columns outside this model\'s feature list.')
        result[name] = list(columns)
    if not result:
        raise ValueError('Choose individual features or at least one feature group.')
    return result


def _shuffle_blocks(frame, shuffle_within):
    # VALIDATION LOGIC: Constrained shuffles require caller-selected context columns, never inferred dates.
    columns = [shuffle_within] if isinstance(shuffle_within, str) else list(shuffle_within or [])
    if len(set(columns)) != len(columns) or set(columns) - set(frame):
        raise ValueError('shuffle_within must name distinct columns present in the supplied data.')
    # CORE LOGIC: STEP 1 — Partition local row positions by explicit permutation context.
    # Input: context=['Mon','Tue','Mon'], shuffle_within=['context'].
    # Output: blocks=[[0,2],[1]], movable_records=2.
    # Explanation: The two Monday feature vectors may swap; the single Tuesday record cannot move.
    # Trick: groupby.indices contains positions, not dataframe index labels; missing contexts form their own group.
    blocks = (list(frame.groupby(columns, dropna=False, observed=True, sort=False).indices.values())
              if columns else [np.arange(len(frame))])
    movable = sum(len(block) for block in blocks if len(block) > 1)
    return columns, blocks, movable


def _shuffled_frame(frame, columns, blocks, rng):
    # CORE LOGIC: STEP 1 — Permute a feature group jointly while preserving its internal row relationships.
    # Input: x=[1,2,3], y=[10,20,30], columns=['x','y'], permutation=[2,0,1].
    # Output: x=[3,1,2], y=[30,10,20]; every x/y pair still comes from one original row.
    # Explanation: Other model inputs stay on their original records, breaking the group's predictive association.
    # Trick: ExtensionArray.take preserves pandas categorical categories; array assignment avoids index alignment.
    order = np.arange(len(frame))
    for block in blocks:
        order[block] = rng.permutation(block)
    shuffled = frame.copy()
    for column in columns:
        shuffled[column] = frame[column].array.take(order)
    return shuffled


def _importance_tables(records, metric, groups):
    # CORE LOGIC: STEP 1 — Aggregate repeated loss changes independently for each feature group.
    # Input: group=['x','x'], mae_delta=[1,3], rmse_delta=[2,4].
    # Output: x has mae_increase_mean=2, mae_increase_std=1.41421356 (approx), repeats=2.
    # Explanation: Positive importance means shuffling worsened held-out prediction accuracy.
    # Trick: Repeat standard deviation uses ddof=1; with one repeat it stays NaN rather than implying certainty.
    repeats = pd.DataFrame(records)
    table = repeats.groupby('group', sort=False, observed=True).agg(
        mae_increase_mean=('mae_delta', 'mean'), mae_increase_std=('mae_delta', 'std'),
        rmse_increase_mean=('rmse_delta', 'mean'), rmse_increase_std=('rmse_delta', 'std'),
        repeats=('repeat', 'size')).reset_index()
    table['feature_columns'] = table['group'].map(lambda name: tuple(groups[name]))
    table['feature_count'] = table['group'].map(lambda name: len(groups[name]))
    table = table.sort_values(f'{metric}_increase_mean', ascending=False, kind='stable').reset_index(drop=True)
    return table, repeats


def permutation_importance(data, actual, model, *, groups=None, include_individual=True,
                           metric='mae', error_scale=1.0, unit='units', max_rows=2000, repeats=5,
                           random_state=0, shuffle_within=None, max_predictions=500,
                           expected_prediction_column=None, progress=None):
    """Measure accuracy lost when inputs are shuffled, using fitted models only.

    Pass a held-out, already filtered population. For two-model paired importance,
    call on the same finite cohort with the same max_rows and random_state.
    ``progress`` receives {stage, completed, total}; raising from it cancels work.
    """
    # CONFIGURATION LOGIC: Every expensive dimension is bounded and caller-visible.
    _validate_model(model, error_scale)
    max_rows = _positive_integer(max_rows, 'max_rows', 50000)
    repeats = _positive_integer(repeats, 'repeats', 100)
    max_predictions = _positive_integer(max_predictions, 'max_predictions', 10000)
    if metric not in {'mae', 'rmse'}:
        raise ValueError("metric must be 'mae' or 'rmse'.")
    groups = _feature_groups(model, groups, include_individual)
    total = 1 + repeats * len(groups)
    if total > max_predictions:
        raise ValueError(f'This request needs {total} prediction calls; reduce groups/repeats or raise max_predictions explicitly.')
    # INFERENCE LOGIC: Sample before inference, then establish one fixed finite baseline cohort.
    frame, positions, target, offset, input_rows, eligible_rows = _input_sample(
        data, model, actual, max_rows, random_state)
    _notify(progress, 'Baseline prediction', 0, total)
    raw = _regression_prediction(frame, model)
    _check_saved_prediction(frame, raw, expected_prediction_column)
    prediction, error = _errors(target, raw, offset, error_scale)
    # CORE LOGIC: STEP 1 — Freeze the finite baseline sample for every permutation repeat.
    # Input: sampled positions=[0,1,2], raw=[2,NaN,4], offset=[10,10,10], actual=[11,12,13].
    # Output: positions=[0,2], prediction=[12,14], error=[1,1], baseline MAE=1.
    # Explanation: Baseline-invalid predictions are excluded once and reported, never reselected after a shuffle.
    # Trick: Identical boolean positions select the dataframe, target and fixed offset together.
    finite = np.isfinite(raw) & np.isfinite(prediction) & np.isfinite(error)
    sampled_rows = len(frame)
    frame, positions = frame.iloc[np.flatnonzero(finite)].copy(), positions[finite]
    target, offset, error = target[finite], offset[finite], error[finite]
    # VALIDATION LOGIC: A singleton or all-singleton constrained population cannot measure feature reliance.
    if len(frame) < 2:
        raise ValueError('At least two finite baseline records are required for permutation importance.')
    contexts, blocks, movable = _shuffle_blocks(frame, shuffle_within)
    if movable < 2:
        raise ValueError('No records can exchange feature values within the requested shuffle groups.')
    baseline = _losses(error)
    records, rng = [], np.random.default_rng(random_state)
    _notify(progress, 'Baseline prediction', 1, total)
    # INFERENCE LOGIC: Sequential prediction keeps memory bounded and allows progress-driven cancellation.
    for label, columns in groups.items():
        for repeat in range(repeats):
            shuffled = _shuffled_frame(frame, columns, blocks, rng)
            changed = _regression_prediction(shuffled, model)
            final, errors = _errors(target, changed, offset, error_scale)
            if not np.isfinite(changed).all() or not np.isfinite(final).all() or not np.isfinite(errors).all():
                raise ValueError(f'Permutation {label!r} produced invalid predictions; evaluation rows were not silently dropped.')
            # CORE LOGIC: STEP 2 — Record final-target losses relative to the fixed baseline.
            # Input: baseline MAE=1, RMSE=2; permuted MAE=3, RMSE=5; repeat=0.
            # Output: repeat=1, mae_delta=2, rmse_delta=3.
            # Explanation: Negative changes mean this particular shuffle helped; they are not truncated to zero.
            # Trick: One shared baseline supports like-for-like changes across groups and repeats.
            losses = _losses(errors)
            records.append(dict(group=label, repeat=repeat + 1, mae=losses['mae'], rmse=losses['rmse'],
                                mae_delta=losses['mae'] - baseline['mae'], rmse_delta=losses['rmse'] - baseline['rmse']))
            _notify(progress, f'Permutation: {label}', 1 + len(records), total)
    # REPORTING LOGIC: Exact evaluated positions make two-model cohort checks and replays possible.
    importance, repetition = _importance_tables(records, metric, groups)
    baseline_table = pd.DataFrame([dict(model=model.name, n=len(frame), **baseline)])
    rows = pd.DataFrame(dict(row_position=positions, source_index=list(frame.index)))
    settings = dict(method='permutation', model=model.name, metric=metric, unit=unit, error_scale=error_scale,
                    max_rows=max_rows, sampled_rows=sampled_rows, input_rows=input_rows, eligible_rows=eligible_rows,
                    evaluated_rows=len(frame), invalid_baseline_predictions=sampled_rows-len(frame), repeats=repeats,
                    random_state=random_state, shuffle_within=contexts, movable_records=movable,
                    prediction_calls=total, offset=model.offset, groups=groups,
                    expected_prediction_column=expected_prediction_column)
    notes = ['Positive loss increase means the fitted model relied on this input on these held-out records.',
             'Repeat SD measures shuffle randomness, not a confidence interval or significance test.',
             'Correlated features may substitute for one another; grouped shuffling keeps within-group relationships.',
             'Shuffling can create implausible inputs. Within-context shuffles answer a conditional reliance question.',
             'Offsets are fixed, including when the same column is also a permuted model feature. No models were fitted.']
    return ExplanationResult(dict(importance=importance, repeats=repetition, baseline=baseline_table, rows=rows), settings, notes)


def _tree_backend(model):
    # VALIDATION LOGIC: A pipeline hides its transformed feature space; do not fabricate raw-column attributions.
    estimator = model.estimator
    if hasattr(estimator, 'steps'):
        raise ValueError('Tree SHAP requires the fitted tree and its prepared feature dataframe, not a preprocessing pipeline. Use permutation for pipelines.')
    module = type(estimator).__module__.split('.')[0]
    if module == 'lightgbm':
        objective = str(getattr(estimator, 'params', {}).get('objective', '')).lower()
        if any(name in objective for name in ['binary', 'multiclass', 'cross_entropy', 'lambdarank', 'rank_xendcg']):
            raise ValueError('Tree SHAP requires a scalar regression model with identity output.')
        return 'lightgbm_native', None
    # SETUP LOGIC: Native LightGBM TreeSHAP needs no extra package; other tree families use optional SHAP.
    try:
        import shap
    except ImportError as exc:
        raise ImportError('Install model-comparison-engine[explain] for non-LightGBM Tree SHAP, or use permutation importance.') from exc
    try:
        explainer = shap.TreeExplainer(estimator, model_output='raw', feature_perturbation='tree_path_dependent')
    except Exception as exc:
        raise ValueError(f'This fitted estimator is not supported by Tree SHAP. Use permutation importance. {exc}') from exc
    return 'shap_tree_path_dependent', explainer


def _tree_values(frame, model, backend, explainer):
    # INFERENCE LOGIC: Native contributions inherit the same iteration defaults as normal LightGBM predict().
    features = frame.loc[:, model.features]
    if backend == 'lightgbm_native':
        values = model.estimator.predict(features, pred_contrib=True)
        values = values.toarray() if hasattr(values, 'toarray') else np.asarray(values)
        if values.shape != (len(frame), len(model.features) + 1):
            raise ValueError('Native Tree SHAP returned multiple outputs or an unexpected contribution shape.')
        return np.asarray(values[:, :-1], dtype=float), np.asarray(values[:, -1], dtype=float)
    # INFERENCE LOGIC: Additivity is checked against this estimator's public predict output below.
    values = np.asarray(explainer.shap_values(features, check_additivity=False), dtype=float)
    expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if values.shape != (len(frame), len(model.features)) or len(expected) != 1:
        raise ValueError('Only one scalar regression output is supported by Tree SHAP.')
    return values, np.full(len(frame), expected[0])


def _shap_global(values, features, error_scale):
    # CORE LOGIC: STEP 1 — Summarize magnitude and direction separately across explained rows.
    # Input: contributions=[[2,-1],[-2,3]], features=['x','y'], error_scale=1.
    # Output: x has mean_abs_shap=2, mean_signed_shap=0; y has mean_abs_shap=2, mean_signed_shap=1.
    # Explanation: A mean signed contribution can cancel even when the feature strongly changes predictions.
    # Trick: Column normalization avoids overflowing a sum of large finite contributions; zero columns remain zero.
    scaled = values * error_scale
    magnitude = np.max(np.abs(scaled), axis=0)
    normalized = np.divide(scaled, magnitude, out=np.zeros_like(scaled), where=magnitude != 0)
    table = pd.DataFrame(dict(feature=features, mean_abs_shap=np.mean(np.abs(normalized), axis=0)*magnitude,
                              mean_signed_shap=np.mean(normalized, axis=0)*magnitude, n=len(values)))
    return table.sort_values('mean_abs_shap', ascending=False, kind='stable').reset_index(drop=True)


def _shap_prediction_table(frame, positions, raw, offset, target, values, base, error_scale, actual):
    # CORE LOGIC: STEP 1 — Expose every component of residual-target reconstruction.
    # Input: base=[2], contributions=[[3,-1]], raw=[4], offset=[100], actual=[103], error_scale=1.
    # Output: shap_sum=2, reconstructed_raw=4, final_prediction=104, reconstructed_final=104, residual=1.
    # Explanation: SHAP explains only the learned model; the caller's fixed offset is a separate reconstruction term.
    # Trick: Never fold the fixed offset into a feature contribution, even if its column is also a model input.
    final, errors = _errors(target, raw, offset, error_scale)
    total = values.sum(axis=1)
    reconstructed = base + total
    table = pd.DataFrame(dict(row_position=positions, source_index=list(frame.index), base_value=base,
                              shap_sum=total, raw_prediction=raw, reconstructed_raw=reconstructed,
                              offset=offset, final_prediction=final, reconstructed_final=reconstructed+offset,
                              additivity_error=reconstructed-raw))
    if actual:
        table['actual'], table['residual'] = target, errors
    return table


def _shap_local_table(frame, predictions, values, features, max_local_rows, error_scale, actual):
    # CORE LOGIC: STEP 1 — Select review cases by final-target absolute error when actuals are supplied.
    # Input: source positions=[4,8,9], residuals=[1,-5,2], max_local_rows=2.
    # Output: local source positions=[8,9]; their per-feature contributions are retained in full.
    # Explanation: Without actual values the first selected rows are retained; importance means still use all sampled rows.
    # Trick: Stable sorting is positional, so source index duplicates and equal error ties cannot misalign explanations.
    chosen = (np.argsort(-np.abs(predictions['residual'].to_numpy()), kind='stable')
              if actual else np.arange(len(predictions)))[:max_local_rows]
    # CORE LOGIC: STEP 2 — Expand selected row-by-feature contributions into a portable long table.
    # Input: position=8, features=['x','y'], feature values=[3,5], SHAP=[2,-1], error_scale=1.
    # Output: [(8,'x',3,2,2),(8,'y',5,-1,-1)] for position, feature, value, SHAP, scaled SHAP.
    # Explanation: Feature values travel beside contributions so a reviewer can inspect the actual model inputs.
    # Trick: iloc accesses original feature values without stringifying or converting categorical values to codes.
    records = []
    for local in chosen:
        for column, feature in enumerate(features):
            records.append(dict(row_position=int(predictions.iloc[local]['row_position']),
                                source_index=frame.index[local], feature=feature, feature_value=frame.iloc[local][feature],
                                shap_value=float(values[local, column]), shap_value_scaled=float(values[local, column]*error_scale)))
    return pd.DataFrame(records)


def tree_shap(data, model, *, actual=None, max_rows=500, max_local_rows=20, row_positions=None,
              error_scale=1.0, unit='units', random_state=0, batch_size=128,
              expected_prediction_column=None, progress=None):
    """Explain fitted tree predictions with native LightGBM or optional TreeExplainer.

    Only scalar identity-output regression is supported. No background data are
    silently drawn from the evaluation set. Attributions use training tree paths.
    For a specific case pass its integer position in the supplied dataframe.
    """
    # CONFIGURATION LOGIC: Sampling and batches bound memory; local tables retain only requested review cases.
    _validate_model(model, error_scale)
    max_rows = _positive_integer(max_rows, 'max_rows', 50000)
    max_local_rows = _positive_integer(max_local_rows, 'max_local_rows', 1000)
    batch_size = _positive_integer(batch_size, 'batch_size', 2000)
    frame, positions, target, offset, input_rows, eligible_rows = _input_sample(
        data, model, actual, max_rows, random_state, row_positions)
    backend, explainer = _tree_backend(model)
    parts, bases = [], []
    total = math.ceil(len(frame) / batch_size) + 1
    _notify(progress, 'SHAP baseline prediction', 0, total)
    raw = _regression_prediction(frame, model)
    _check_saved_prediction(frame, raw, expected_prediction_column)
    _notify(progress, 'SHAP baseline prediction', 1, total)
    # INFERENCE LOGIC: Existing trees are explained in bounded batches; no estimator is fitted or refitted.
    for number, start in enumerate(range(0, len(frame), batch_size), start=1):
        values, base = _tree_values(frame.iloc[start:start+batch_size], model, backend, explainer)
        parts.append(values)
        bases.append(base)
        _notify(progress, 'Tree SHAP', number+1, total)
    # CORE LOGIC: STEP 1 — Reassemble batch outputs without changing the sampled row order.
    # Input: batch SHAP=[[[1,2]], [[3,4]]], batch bases=[[10],[10]].
    # Output: SHAP=[[1,2],[3,4]], bases=[10,10], reconstructed raw predictions=[13,17].
    # Explanation: Concatenation uses the same batch order as the original selected feature rows.
    # Trick: A numerical additivity check below rejects transforms or tree limits that differ from predict().
    values, base = np.concatenate(parts, axis=0), np.concatenate(bases)
    reconstructed = base + values.sum(axis=1)
    # VALIDATION LOGIC: Do not claim an explanation of the supplied prediction when the outputs differ.
    if not all(np.isfinite(array).all() for array in [raw, values, base, reconstructed]):
        raise ValueError('SHAP or model predictions contain nonfinite values on the selected rows.')
    if not np.allclose(reconstructed, raw, rtol=1e-5, atol=1e-6):
        raise ValueError('SHAP base + contributions do not match model.predict(). A nonlinear output transform or different tree limit may be present; use permutation importance.')
    if not np.isfinite(values * error_scale).all() or not np.isfinite(raw + offset).all():
        raise ValueError('The requested scale or reconstruction offset overflows on these explanations.')
    # REPORTING LOGIC: Tables expose raw and final scales explicitly instead of conflating SHAP with accuracy.
    predictions = _shap_prediction_table(frame, positions, raw, offset, target, values, base, error_scale, actual)
    if actual and not np.isfinite(predictions['residual']).all():
        raise ValueError('The requested error scale overflows on the final-target residuals.')
    local = _shap_local_table(frame, predictions, values, model.features, max_local_rows, error_scale, actual)
    global_table = _shap_global(values, model.features, error_scale)
    settings = dict(method='tree_shap', backend=backend, model=model.name, actual=actual, unit=unit,
                    error_scale=error_scale, input_rows=input_rows, eligible_rows=eligible_rows, evaluated_rows=len(frame),
                    max_rows=max_rows, max_local_rows=max_local_rows, random_state=random_state, offset=model.offset,
                    perturbation='tree_path_dependent', additivity_rtol=1e-5, additivity_atol=1e-6,
                    expected_prediction_column=expected_prediction_column,
                    row_selection='explicit_positions' if row_positions is not None else 'seeded_sample_or_all')
    notes = ['SHAP explains the fitted prediction, not accuracy improvement or a causal effect.',
             'Path-dependent expectations use the fitted tree structure and its training path frequencies.',
             'Global contributions use all explained rows; local tables select largest final-target errors when actuals are provided.',
             'Contributions in local shap_value and prediction reconstruction use native target units; scaled columns use error_scale.',
             'The offset is fixed per record and shown separately from the learned-model contributions. No models were fitted.']
    if row_positions is not None:
        notes.append('Explicitly selected cases are not a representative sample; their global SHAP means describe only these cases.')
    return ExplanationResult(dict(global_importance=global_table, local_contributions=local, predictions=predictions), settings, notes)
