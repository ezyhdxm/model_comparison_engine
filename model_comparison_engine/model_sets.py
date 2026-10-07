"""Named prediction/model collections with cached inference and explicit pair selection."""
# SETUP LOGIC: Estimators remain caller-owned, in memory, and are never serialized or fitted.
from .data import read_data
import pandas as pd


def compare_prediction_set(data, actual, predictions, *, reference=None, candidate=None,
                           prediction_offsets=None, **kwargs):
    """Compare a selected pair from a mapping of model names to saved prediction columns."""
    # VALIDATION LOGIC: Model names, columns and offsets must be unambiguous.
    from .engine import compare_predictions
    columns, offsets = dict(predictions), dict(prediction_offsets or {})
    if len(columns) < 2 or any(not isinstance(n, str) or not n.strip() for n in columns):
        raise ValueError('Supply at least two nonempty model names mapped to prediction columns.')
    if len(set(columns.values())) != len(columns):
        raise ValueError('Each named model needs its own prediction column.')
    if set(offsets)-set(columns):
        raise ValueError('Offset names must identify models in predictions.')
    frame = data if isinstance(data,pd.DataFrame) else read_data(data,
        string_columns=[kwargs.get('id_column'), kwargs.get('entity_column')])
    if set(columns.values())-set(frame) or any(v is not None and v not in frame for v in offsets.values()):
        raise ValueError('A prediction or offset column is missing from data.')
    reference = next(iter(columns)) if reference is None else reference
    candidate = next((n for n in columns if n != reference), None) if candidate is None else candidate
    if reference not in columns or candidate not in columns or reference == candidate:
        raise ValueError('Select two distinct names from predictions.')
    # CONFIGURATION LOGIC: Pair metadata stays attached to each model when the roles change.
    reserved = {'reference_name','candidate_name','reference_offset','candidate_offset'}
    if reserved & set(kwargs):
        raise ValueError('Use named predictions and prediction_offsets to configure model identities.')
    result = compare_predictions(frame, actual, columns[reference], columns[candidate],
        reference_name=reference, candidate_name=candidate,
        reference_offset=offsets.get(reference), candidate_offset=offsets.get(candidate), **kwargs)
    result.prediction_columns = columns
    result.prediction_offsets = {n: offsets.get(n) for n in columns}
    return result


def compare_model_set(data, actual, models, *, reference=None, candidate=None, **kwargs):
    """Predict once per fitted model, retaining all models for pair switching and explanations."""
    # VALIDATION LOGIC: Require explicit independent feature schemas and unique names.
    from .engine import Model, _predict
    models = list(models)
    if len(models) < 2 or any(not isinstance(m, Model) for m in models):
        raise TypeError('Supply at least two fitted estimators wrapped in Model objects.')
    names = [m.name for m in models]
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError('Models must have distinct nonempty string names.')
    if (reference is not None and reference not in names) or (candidate is not None and candidate not in names):
        raise ValueError('The requested reference/candidate name is not in models.')
    if reference is not None and reference == candidate:
        raise ValueError('Choose different reference and candidate models.')
    frame = read_data(data, string_columns=[kwargs.get('id_column'), kwargs.get('entity_column')])
    columns = {name: f'Model prediction {i + 1}' for i, name in enumerate(names)}
    if set(columns.values()) & set(frame):
        raise ValueError('Rename existing Model prediction N columns before calling compare_model_set.')
    # INFERENCE LOGIC: The registry caches model outputs; selecting another pair calls no estimator.
    for model in models:
        frame[columns[model.name]] = _predict(frame, model)
    result = compare_prediction_set(frame, actual, columns, reference=reference, candidate=candidate,
        prediction_offsets={m.name:m.offset for m in models}, **kwargs)
    result.models = {m.name:Model(m.name,m.estimator,list(m.features),m.offset) for m in models}
    return result
