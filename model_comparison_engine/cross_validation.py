"""Optional fold-local fitting for any two scalar regressors with explicit factories."""
# SETUP LOGIC: Existing prediction-only APIs remain independent of this explicit training entry point.
from dataclasses import dataclass
import json
import weakref
import numpy as np
import pandas as pd
from .data import check_key, numeric, read_data
from .engine import compare_predictions
from .walk_forward import WalkForwardConfig, walk_forward_splits


@dataclass
class TrainableModel:
    """Create fresh unfitted objects per fold; optional transformers fit on training rows only.

    ``factory()`` returns an estimator implementing fit(X, y) and predict(X).
    ``transformer_factory()`` optionally returns fit(X, y)/transform(X); a complete
    preprocessing pipeline may instead be returned by factory. No categories,
    imputation, scaling or feature construction are inferred by this package.
    """
    # CONFIGURATION LOGIC: Each model declares its own ordered inputs and additive prediction-time offset.
    name: str
    factory: object
    features: list
    transformer_factory: object = None
    offset: str = None


@dataclass
class WalkForwardResult:
    # RESULT LOGIC: All numerical evidence comes from the actual held-out fold predictions.
    comparison: object
    predictions: pd.DataFrame
    folds: pd.DataFrame
    fold_metrics: pd.DataFrame
    configuration: dict

    def export(self, folder, *, slices=None, interactions=None, min_count=30, metric='mae_delta',
               inference=None, include_candidate=True, candidate_population='candidate', candidate_top_n=20):
        """Export the ordinary paired review plus fold boundaries, exact OOF rows and fold metrics."""
        # FILE IO LOGIC: Extend the newly created immutable review directory; estimator objects are not serialized.
        output = self.comparison.export(folder, slices=slices, interactions=interactions,
                                        min_count=min_count, metric=metric, inference=inference,
                                        include_candidate=include_candidate, candidate_population=candidate_population,
                                        candidate_top_n=candidate_top_n)
        self.folds.to_csv(output/'folds.csv', index=False)
        self.fold_metrics.to_csv(output/'fold_metrics.csv', index=False)
        self.predictions.to_parquet(output/'oof_predictions.parquet', index=False)
        (output/'walk_forward.json').write_text(json.dumps(self.configuration, indent=2, allow_nan=False), encoding='utf-8')
        receipt = output/'review.json'
        metadata = json.loads(receipt.read_text(encoding='utf-8'))
        metadata.update(training_performed=True, evaluation_design='walk_forward_cross_validation',
                        walk_forward_configuration='walk_forward.json', final_holdout_evaluated=False)
        receipt.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
        # REPORTING LOGIC: Use computed tables; pooled row-weighted loss and per-fold loss answer different questions.
        table = self.fold_metrics[['fold_id', 'n', 'reference_mae', 'candidate_mae', 'mae_delta', 'mae_improvement_pct']]
        section = '<section><h2>Walk-forward validation</h2><p>Each record is predicted once by models fitted on earlier dates. '
        section += 'Pooled metrics weight records; the table below shows each fold separately. No final holdout is evaluated. '
        section += 'These folds are model-development evidence, not an untouched final test. '
        section += 'Shared training history and serial dependence can violate the independent-unit assumption in the optional loss tests.</p>'
        section += '<p><a href="folds.csv">Fold boundaries</a> · <a href="fold_metrics.csv">Exact fold metrics</a> · '
        section += '<a href="oof_predictions.parquet">Out-of-fold predictions</a> · <a href="walk_forward.json">Configuration</a></p>'
        section += table.head(100).to_html(index=False, escape=True)+ '</section>'
        report = output/'report.html'
        report.write_text(report.read_text(encoding='utf-8').replace('</body>', section+'</body>'), encoding='utf-8')
        return output


def _definition(model, frame, forbidden):
    # VALIDATION LOGIC: Accept explicit factories rather than mutating caller-owned already-fitted estimators.
    if not isinstance(model, TrainableModel):
        raise TypeError('Use TrainableModel(name, factory, features, transformer_factory=None, offset=None).')
    if not isinstance(model.name, str) or not model.name.strip() or not callable(model.factory):
        raise ValueError('Each model needs a nonempty name and a callable factory returning a fresh unfitted estimator.')
    if isinstance(model.features, str) or not model.features or len(set(model.features)) != len(model.features):
        raise ValueError('Each model needs a nonempty unique ordered feature list.')
    if set(model.features)-set(frame):
        raise ValueError(f'{model.name}: missing feature columns {sorted(set(model.features)-set(frame))}.')
    if set(model.features) & forbidden:
        raise ValueError(f'{model.name}: labels and label-availability timestamps cannot be model inputs.')
    if model.offset is not None and (model.offset not in frame or model.offset in forbidden):
        raise ValueError(f'{model.name}: offset must be an existing prediction-time column, separate from outcomes.')
    if model.transformer_factory is not None and not callable(model.transformer_factory):
        raise TypeError('transformer_factory must be callable or None.')


def _fresh(factory, registry, label):
    # OBJECT LIFECYCLE LOGIC: Weak references catch reused estimators without retaining every fitted model in memory.
    value = factory()
    previous = registry.get(id(value))
    if previous is not None and previous() is value:
        raise ValueError(f'{label}: factory reused an object; return a fresh unfitted object for every fold/model.')
    try:
        registry[id(value)] = weakref.ref(value)
    except TypeError:
        registry[id(value)] = lambda value=value: value
    if not callable(getattr(value, 'fit', None)):
        raise TypeError(f'{label}: factory output must implement fit(X, y).')
    return value


def _check_transformed(values, source, label):
    # VALIDATION LOGIC: Transformers may return arrays/sparse matrices, but cannot silently reorder pandas rows.
    if not hasattr(values, 'shape') or len(values.shape) != 2 or values.shape[0] != len(source):
        raise ValueError(f'{label}: transformer must preserve row count and return a two-dimensional feature matrix.')
    if isinstance(values, pd.DataFrame) and not values.index.equals(source.index):
        raise ValueError(f'{label}: transformed pandas row index differs from its input.')


def _fit_predict(model, train, validation, target, registry):
    # CONFIGURATION LOGIC: Preserve each model's declared source-column order; preprocessing is caller-defined.
    train_x, validation_x = train.loc[:, model.features], validation.loc[:, model.features]
    estimator = _fresh(model.factory, registry, model.name)
    if not callable(getattr(estimator, 'predict', None)):
        raise TypeError(f'{model.name}: estimator must implement predict(X).')
    # CORE LOGIC: STEP 1 — Fit an optional preprocessing object only on this fold's training records.
    # Input: training x=[2,4], validation x=[100]; factory makes a mean-centering transformer.
    # Output: fit sees [2,4], learned mean=3; transformed training=[-1,1], validation=[97].
    # Explanation: A category encoder, imputer or scaler can learn only from the past training population.
    # Trick: The validation target is never passed to fit; transformer output row order is checked before fitting.
    if model.transformer_factory is not None:
        transformer = _fresh(model.transformer_factory, registry, model.name+' transformer')
        if not callable(getattr(transformer, 'transform', None)):
            raise TypeError('A transformer must implement transform(X).')
        transformer.fit(train_x, target)
        train_x, validation_x = transformer.transform(train_x), transformer.transform(validation_x)
        _check_transformed(train_x, train, model.name)
        _check_transformed(validation_x, validation, model.name)
    # MODELING LOGIC: Arbitrary fit/predict protocol; no model defaults, feature encoding or tuning is inserted.
    estimator.fit(train_x, target)
    prediction = estimator.predict(validation_x)
    # VALIDATION LOGIC: Require one scalar per held-out row, with no pandas index misalignment.
    if isinstance(prediction, (pd.Series, pd.DataFrame)) and not prediction.index.equals(validation.index):
        raise ValueError(f'{model.name}: pandas predictions must preserve validation row order/index.')
    values = np.asarray(prediction)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) != len(validation):
        raise ValueError(f'{model.name}: predict must return one scalar per validation record.')
    return values


def _comparison_options(reference, candidate, actual, time_column, id_column, entity_column, timezone,
                        error_scale, unit, tolerance):
    # CONFIGURATION LOGIC: Existing offset and metric semantics remain identical in pooled and per-fold reviews.
    return dict(actual=actual, reference='cv_reference_prediction', candidate='cv_candidate_prediction',
                reference_name=reference.name, candidate_name=candidate.name, id_column=id_column,
                time_column=time_column, entity_column=entity_column, timezone=timezone,
                error_scale=error_scale, unit=unit, tolerance=tolerance,
                reference_offset=reference.offset, candidate_offset=candidate.offset)


def _definition_metadata(model):
    # PROVENANCE LOGIC: Record declarative schemas; caller factories and estimator implementations remain external code.
    factory = model.factory
    label = lambda item: str(getattr(item, '__module__', None) or 'user')+'.'+str(getattr(item, '__qualname__', type(item).__qualname__))
    return dict(name=model.name, features=list(model.features), offset=model.offset, factory=label(factory),
                transformer_factory=label(model.transformer_factory) if model.transformer_factory is not None else None)


def walk_forward_compare(data, target, reference, candidate, *, time_column, settings=None, actual=None,
                         id_column=None, entity_column=None, timezone='UTC', error_scale=1., unit='units',
                         tolerance=1., min_count=30, progress=None):
    """Explicitly fit two fresh regressors per fold and compare disjoint out-of-fold predictions.

    The target is shared by both models; actual defaults to target and may instead
    specify an observed level when offsets restore predicted deltas. All source
    features must already be causal. Never pass test data inside a model factory.
    """
    # VALIDATION LOGIC: Preserve source metadata for arbitrary downstream slices; reserve only new CV output names.
    frame = read_data(data, string_columns=[id_column, entity_column])
    settings = settings if isinstance(settings, WalkForwardConfig) else WalkForwardConfig(**(settings or {}))
    actual = actual or target
    reserved = {'cv_row_position', 'cv_fold', 'cv_reference_prediction', 'cv_candidate_prediction'}
    if frame.columns.duplicated().any() or reserved & set(frame) or any(str(c).startswith('__') for c in frame):
        raise ValueError('Provide unique columns without reserved cv_* result names or __ prefixes.')
    if target not in frame or actual not in frame:
        raise ValueError('Target and actual must name existing source columns.')
    if id_column is not None:
        check_key(frame, id_column)
    forbidden = {target, actual, settings.label_available_column}
    for model in [reference, candidate]:
        _definition(model, frame, forbidden)
    if reference.name == candidate.name:
        raise ValueError('Use distinct model names.')
    # ORCHESTRATION LOGIC: Plan all dates before any fitting; fold positions never include the final reservation.
    folds = walk_forward_splits(frame, time_column, settings, timezone=timezone)
    options = _comparison_options(reference, candidate, actual, time_column, id_column, entity_column,
                                  timezone, error_scale, unit, tolerance)
    compare_predictions(frame.iloc[:0].assign(cv_reference_prediction=0., cv_candidate_prediction=0.), **options)
    predictions, summaries, registry = [], [], {}
    for fold in folds:
        predicted, summary = _run_fold(frame, fold, target, reference, candidate, registry,
                                        options, min_count, progress, len(folds))
        predictions.append(predicted)
        summaries.append(summary)
    # CORE LOGIC: STEP 1 — Pool disjoint held-out records in original source-row order.
    # Input: fold1 cv_row_position=[3,1], fold2 cv_row_position=[4,2].
    # Output: pooled cv_row_position=[1,2,3,4]; each record contributes once to the common-sample comparison.
    # Explanation: Fold windows do not overlap; sorting restores source order without averaging predictions.
    # Trick: Pooled MAE weights records, whereas a mean of fold MAEs weights folds; neither substitutes for the other.
    pooled = pd.concat(predictions, ignore_index=True).sort_values('cv_row_position', kind='stable').reset_index(drop=True)
    if pooled.cv_row_position.duplicated().any():
        raise RuntimeError('Walk-forward validation rows overlap; no pooled comparison was produced.')
    comparison = compare_predictions(pooled, **options)
    # PROVENANCE LOGIC: Explicitly distinguish development validation, incomplete dates and reserved final evaluation.
    configuration = dict(settings=settings.to_dict(), target=target, actual=actual, time_column=time_column,
        timezone=timezone, reference=_definition_metadata(reference), candidate=_definition_metadata(candidate),
        input_rows=len(frame), oof_rows=len(pooled), unscored_rows=len(frame)-len(pooled), folds=len(folds),
        final_holdout_evaluated=False, labels_assumed_available_at_record_time=settings.label_available_column is None,
        scoring='Retrospective OOF; label availability gates training, not the later score computation. No model is selected.',
        note='Unscored rows include initial training, gaps, partial final blocks and reserved holdout/buffer. '
             'Before model selection, exclude OOF outcomes unavailable at the intended selection cutoff. '
             'Factories must return unfitted objects; causal features and factory code remain caller responsibilities.')
    return WalkForwardResult(comparison, pooled, pd.DataFrame([fold.to_dict() for fold in folds]),
                             pd.DataFrame(summaries), configuration)


def _run_fold(frame, fold, target, reference, candidate, registry, options, min_count, progress, total):
    # CORE LOGIC: STEP 1 — Use the same finite-label training records for both model definitions.
    # Input: scheduled train row positions=[0,1,2], targets=[3,NaN,5]; validation positions=[4,5].
    # Output: train rows=[0,2], train labels=[3.,5.], validation rows=[4,5] regardless of their target availability.
    # Explanation: Invalid training labels cannot fit either estimator; held-out unknown outcomes remain coverage gaps.
    # Trick: Positional source selection and float64 labels prevent index alignment or integer overflow differences.
    scheduled = frame.iloc[fold.train_positions].copy()
    labels = numeric(scheduled, target)
    finite = np.isfinite(labels.to_numpy())
    train = scheduled.iloc[np.flatnonzero(finite)].copy()
    labels = labels.iloc[np.flatnonzero(finite)]
    validation = frame.iloc[fold.validation_positions].copy()
    if len(train) < 2:
        raise ValueError(f'Fold {fold.fold_id} has fewer than two finite available training labels.')
    # MODELING LOGIC: Each call gets fresh estimator/preprocessor state and fixed model-specific feature columns.
    for position, (model, column) in enumerate([(reference, 'cv_reference_prediction'), (candidate, 'cv_candidate_prediction')]):
        if progress is not None:
            progress('walk_forward', (fold.fold_id-1)*2+position, total*2, f'Fold {fold.fold_id}/{total}: fitting {model.name}')
        validation[column] = _fit_predict(model, train, validation, labels, registry)
    # REPORTING LOGIC: Attach generated fold identity only after models have made their held-out predictions.
    validation['cv_row_position'], validation['cv_fold'] = fold.validation_positions, fold.fold_id
    summary = compare_predictions(validation, **options).summary(min_count=min_count).iloc[0].to_dict()
    summary.update(fold_id=fold.fold_id, train_finite_target_rows=len(train), validation_input_rows=len(validation))
    if progress is not None:
        progress('walk_forward', fold.fold_id*2, total*2, f'Fold {fold.fold_id}/{total}: complete')
    return validation, summary
