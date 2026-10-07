"""Public paired-model API for scalar regression targets."""
# SETUP LOGIC: These paired-review APIs use existing predictions; optional cross-validation lives separately.
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from .data import paired_rows, candidate_rows as prepare_candidate_rows, read_data
from .metrics import metrics, grouped_metrics, date_sensitivity
from .slices import Slice, slice_labels, default_slices
from .inference import InferenceConfig, grouped_tests


class Comparison:
    # SETUP LOGIC: Retain source inputs and explicit configuration for reproducible filtered reviews.
    def __init__(self, data, rows, coverage, config):
        self.data, self.rows, self.coverage, self.config = data, rows, coverage, config
        self.reference_name, self.candidate_name = config['reference_name'], config['candidate_name']
        self.unit, self.tolerance = config['unit'], config['tolerance']
        self.default_slices = default_slices(rows.columns) if config['time_column'] is not None else []
        self.filter_history = []
        self.models, self.explanations = {}, {}
        self.prediction_columns = {self.reference_name:config['reference'], self.candidate_name:config['candidate']}
        self.prediction_offsets = {self.reference_name:config['reference_offset'], self.candidate_name:config['candidate_offset']}

    def select_models(self, reference, candidate):
        """Select two cached predictions; no inference or training is performed."""
        # CONFIGURATION LOGIC: Preserve metadata and named offsets while changing only the selected pair.
        from .model_sets import compare_prediction_set
        options = {k:v for k,v in self.config.items() if k not in
                   {'actual','reference','candidate','reference_name','candidate_name','reference_offset','candidate_offset'}}
        result = compare_prediction_set(self.data, self.config['actual'], self.prediction_columns,
            reference=reference, candidate=candidate, prediction_offsets=self.prediction_offsets, **options)
        result.models, result.filter_history = dict(self.models), list(self.filter_history)
        return result

    def _explanation_input(self, model, population):
        # VALIDATION LOGIC: Saved predictions cannot identify model internals or simulate feature perturbations.
        name = self.candidate_name if model is None else model
        if name not in self.models:
            raise ValueError('Explanations require a fitted model. Use compare_models or compare_model_set first.')
        fitted = self.models[name]
        if fitted.name != name or fitted.offset != self.prediction_offsets.get(name):
            raise ValueError('Model identity or offset changed after prediction. Rebuild the comparison before explaining it.')
        if population not in {'paired','model'}:
            raise ValueError("Use population='paired' or 'model' for explanations.")
        # CORE LOGIC: STEP 1 — Recover feature rows by position, preserving duplicate input indexes safely.
        # Input: data index=[7,7,9], x=[10,20,30], paired source positions=[0,2], population='paired'.
        # Output: selected x=[10,30], index=[7,9]; x=20 is not accidentally reintroduced by a label join.
        # Explanation: Paired explanations use the same finite actual and two predictions as the error comparison.
        # Trick: iloc uses source positions; the model population lets the explanation function audit its own rows.
        frame = self.data.iloc[self.rows['__source_position']] if population == 'paired' else self.data
        return name, self.models[name], frame

    def _explanation_options(self, name, options):
        # VALIDATION LOGIC: Comparison-bound explanations use exactly the same declared unit and reconstruction audit.
        fixed = dict(error_scale=self.config['error_scale'], unit=self.unit,
                     expected_prediction_column=self.prediction_columns[name])
        for key, value in fixed.items():
            if key in options and options[key] != value:
                raise ValueError(f'{key} must match the applied comparison; use the standalone explanation API to change it.')
            options[key] = value
        return options

    def _record_explanation(self, result, population):
        # CONFIGURATION LOGIC: Public row_position remains relative to the explicitly selected explanation cohort.
        positions = self.rows['__source_position'].to_numpy() if population == 'paired' else np.arange(len(self.data))
        identity = self.config['id_column']
        # CORE LOGIC: STEP 1 — Attach an unambiguous link back to the applied input table.
        # Input: paired source positions=[1,3], explanation row_position=[1], data record IDs=['a','b','c','d'].
        # Output: comparison_source_position=[3], record_id=['d']; explanation row_position stays [1].
        # Explanation: Feature-level tables may repeat a position; every contribution gets the same source identity.
        # Trick: NumPy positional lookup is safe when original pandas index labels are duplicated.
        for table in result.tables.values():
            if 'row_position' in table:
                source = positions[table['row_position'].to_numpy(dtype=int)]
                table['comparison_source_position'] = source
                if identity is not None:
                    table['record_id'] = self.data.iloc[source][identity].to_numpy()
        # REPORTING LOGIC: Sampling positions and record keys remain interpretable outside the live notebook.
        result.settings['record_id_column'] = identity
        result.settings['position_scope'] = 'row_position: explanation cohort; comparison_source_position: applied comparison.data'
        return result

    def permutation_importance(self, model=None, *, population='paired', **kwargs):
        """Measure held-out loss changes for a retained model; results are included in later exports."""
        # EXPLANATION LOGIC: Expensive repeated predictions happen only on explicit request.
        from .explainability import permutation_importance
        name, fitted, frame = self._explanation_input(model, population)
        kwargs = self._explanation_options(name, kwargs)
        result = permutation_importance(frame, self.config['actual'], fitted, **kwargs)
        self._record_explanation(result, population)
        result.settings.update(population=population, filters=list(self.filter_history), unit=self.unit)
        self.explanations[f'permutation:{name}'] = result
        return result

    def shap_values(self, model=None, *, population='paired', **kwargs):
        """Explain retained fitted tree predictions, keeping any additive offset separate."""
        # EXPLANATION LOGIC: Verify contribution additivity before presenting local or aggregate explanations.
        from .explainability import tree_shap
        name, fitted, frame = self._explanation_input(model, population)
        kwargs = self._explanation_options(name, kwargs)
        result = tree_shap(frame, fitted, actual=self.config['actual'], **kwargs)
        self._record_explanation(result, population)
        result.settings.update(population=population, filters=list(self.filter_history), unit=self.unit)
        self.explanations[f'shap:{name}'] = result
        return result

    def trade_diagnostics(self, settings=None):
        # REPORTING LOGIC: Intraday metrics use all paired records; plotting samples are reported separately.
        from .trade_view import trade_tables
        return trade_tables(self, settings)

    def summary(self, min_count=30):
        # REPORTING LOGIC: Reuse the same metric definitions at every aggregation level.
        return pd.DataFrame([metrics(self.rows, tolerance=self.tolerance, min_count=min_count)])

    def slice(self, column, bins=None, labels=None, min_count=30, top_n=20, right=True):
        # CONFIGURATION LOGIC: Accept either a reusable Slice or simple keyword arguments.
        spec = column if isinstance(column, Slice) else Slice(column, bins, labels, right, top_n)
        # CORE LOGIC: STEP 1 — Attach groups to paired records, then recompute per-group losses.
        # Input: segment=['A','B'], reference abs=[2,4], candidate abs=[1,2].
        # Output: group A has n=1, MAE=(2,1); group B has n=1, MAE=(4,2).
        # Explanation: Slice membership is defined from data columns, independently of prediction gains.
        # Trick: Copying preserves the shared comparison rows when the user changes slice controls.
        rows = self.rows.assign(group=slice_labels(self.rows, spec))
        return grouped_metrics(rows, ['group'], tolerance=self.tolerance, min_count=min_count)

    def cross_slice(self, x, y, x_bins=None, y_bins=None, min_count=30, top_n=12,
                    x_right=True, y_right=True):
        # CONFIGURATION LOGIC: Both axes may use distinct fixed bins or categorical limits.
        sx = x if isinstance(x, Slice) else Slice(x, x_bins, right=x_right, top_n=top_n)
        sy = y if isinstance(y, Slice) else Slice(y, y_bins, right=y_right, top_n=top_n)
        if sx.column == sy.column:
            raise ValueError('Choose two different slice columns.')
        # CORE LOGIC: STEP 1 — Compute the intersection on records, not on pre-averaged one-way metrics.
        # Input: segment=['A','A','B'], size=['big','small','big'], ref_abs=[2,4,6], cand_abs=[1,3,2].
        # Output: (A,big): MAE=(2,1), n=1; (A,small): (4,3), n=1; (B,big): (6,2), n=1.
        # Explanation: Each combination gets its actual records; (B,small) has no observations.
        # Trick: Empty heatmap cells remain unavailable, rather than appearing as perfect zero error.
        rows = self.rows.assign(x=slice_labels(self.rows, sx), y=slice_labels(self.rows, sy))
        return grouped_metrics(rows, ['x','y'], tolerance=self.tolerance, min_count=min_count)

    def daily(self, min_count=30):
        # CORE LOGIC: STEP 1 — Group ISO dates chronologically, retaining unknown dates separately.
        # Input: date=['2026-04-02','2026-04-01','2026-04-02'], ref_abs=[3,1,5], cand_abs=[2,0,4].
        # Output: Apr 1 has n=1, MAE=(1,0); Apr 2 has n=2, MAE=(4,3), in this date order.
        # Explanation: ISO date strings sort chronologically, regardless of each day's record count.
        # Trick: Date plots never reuse the frequency-ranked top-category display or collapse dates into Other.
        rows = self.rows.assign(group=self.rows['__date'].fillna('Unknown date'))
        return grouped_metrics(rows,['group'],tolerance=self.tolerance,min_count=min_count)

    def stability(self):
        # REPORTING LOGIC: Return descriptive leave-one-date-out sensitivity without a training call.
        return date_sensitivity(self.rows)

    def inference_config(self, inference=None):
        # CONFIGURATION LOGIC: Resolve auto once; an explicit date/entity test requires its mapped column.
        config = InferenceConfig() if inference is None else inference
        if isinstance(config, dict):
            config = InferenceConfig(**config)
        if not isinstance(config, InferenceConfig):
            raise TypeError('inference must be InferenceConfig, a configuration dict or None.')
        if config.unit == 'auto':
            config = replace(config, unit='date' if self.config['time_column'] is not None else 'record')
        for unit, column in [('date', 'time_column'), ('entity', 'entity_column')]:
            if config.unit == unit and self.config[column] is None:
                raise ValueError(f'Configure {column} before using {unit}-level inference.')
        return config

    def paired_test(self, inference=None, min_count=30):
        # INFERENCE LOGIC: Test paired losses using the same explicit policy as the slice heatmaps.
        return grouped_tests(self.rows, [], self.inference_config(inference), min_count=min_count)

    def slice_test(self, column, bins=None, labels=None, min_count=30, top_n=20, right=True, inference=None):
        # CONFIGURATION LOGIC: Keep bin boundaries and top-category choices identical to descriptive slices.
        spec = column if isinstance(column, Slice) else Slice(column, bins, labels, right, top_n)
        # CORE LOGIC: STEP 1 — Preserve row-level pairing before forming each slice's test population.
        # Input: group=['A','A','B'], reference abs=[2,4,3], candidate abs=[1,2,4].
        # Output: A passes paired loss differences [-1,-2]; B passes [1] to the configured inference policy.
        # Explanation: Each observed group is tested separately; adjustment covers eligible tests in this table.
        # Trick: Group labels use exactly the same binning as slice(); no outcome-ranked regrouping is performed.
        rows = self.rows.assign(group=slice_labels(self.rows, spec))
        return grouped_tests(rows, ['group'], self.inference_config(inference), min_count=min_count)

    def cross_slice_test(self, x, y, x_bins=None, y_bins=None, min_count=30, top_n=12,
                         x_right=True, y_right=True, inference=None):
        # CONFIGURATION LOGIC: Reusable Slice objects retain their own independent axis boundaries.
        sx = x if isinstance(x, Slice) else Slice(x, x_bins, right=x_right, top_n=top_n)
        sy = y if isinstance(y, Slice) else Slice(y, y_bins, right=y_right, top_n=top_n)
        if sx.column == sy.column:
            raise ValueError('Choose two different slice columns.')
        # CORE LOGIC: STEP 1 — Pass each observed intersection to the paired inference policy.
        # Input: x=['A','A','B'], y=['big','small','big'], paired loss differences=[-1,2,-3].
        # Output: (A,big) receives [-1]; (A,small) receives [2]; (B,big) receives [-3].
        # Explanation: Empty B/small has no invented observations and therefore no p-value.
        # Trick: Multiple-testing adjustment is applied across eligible observed cells, not across missing grid cells.
        rows = self.rows.assign(x=slice_labels(self.rows, sx), y=slice_labels(self.rows, sy))
        return grouped_tests(rows, ['x','y'], self.inference_config(inference), min_count=min_count)

    def candidate_rows(self, population='candidate'):
        # CONFIGURATION LOGIC: Own-population diagnosis is explicit and never changes paired comparisons.
        if population == 'paired':
            return self.rows.copy()
        if population != 'candidate':
            raise ValueError("population must be 'candidate' or 'paired'.")
        options = {k:self.config[k] for k in ['actual','candidate','time_column','entity_column',
                   'error_scale','timezone','candidate_offset']}
        return prepare_candidate_rows(self.data, **options)

    def candidate_diagnostics(self, slices=None, *, population='candidate', min_count=30, top_n=20, bins=10):
        # DIAGNOSTIC LOGIC: Reuse the user's explicit slices without discarding unsupported or difficult records.
        from .diagnostics import candidate_tables
        specs = self.default_slices if slices is None else slices
        tables = candidate_tables(self.candidate_rows(population), specs, min_count=min_count,
                                  tolerance=self.tolerance, top_n=top_n, bins=bins, id_column=self.config['id_column'])
        tables['summary']['population'] = population
        return tables

    def temporal_diagnostics(self, settings=None, *, population='candidate', entity=None):
        """Describe residual timing using fixed-clock bins and separate within-entity event lags."""
        # CONFIGURATION LOGIC: Time diagnostics are explicit, use existing predictions, and fit no forecasting model.
        from .temporal import TemporalConfig, temporal_tables
        from .event_lags import event_lag_table
        settings = TemporalConfig() if settings is None else settings
        if isinstance(settings, dict):
            settings = TemporalConfig(**settings)
        if not isinstance(settings, TemporalConfig):
            raise TypeError('settings must be TemporalConfig, a configuration dict or None.')
        if entity is not None and self.config['entity_column'] is None:
            raise ValueError('Map entity_column before selecting one temporal entity.')
        # CORE LOGIC: STEP 1 — Restrict the diagnostic series to one explicit entity when requested.
        # Input: entities=['A','B','A'], residuals=[1,9,3], entity='A'.
        # Output: retained entities=['A','A'], residuals=[1,3]; the B prediction is absent from both temporal clocks.
        # Explanation: A single-entity review can separate residual persistence from changing entity composition.
        # Trick: Filtering affects this diagnostic only; the parent paired comparison and original predictions stay intact.
        rows = self.candidate_rows(population)
        rows = rows.loc[rows['__entity'].eq(entity)].copy() if entity is not None else rows
        # DIAGNOSTIC LOGIC: Both clocks reuse the selected errors and retain their own explicit support audits.
        tables = temporal_tables(rows, settings)
        tables['event_autocorrelation'] = event_lag_table(rows, settings.max_lag, settings.signal)
        # REPORTING LOGIC: Preserve population identity with temporal evidence and exported settings.
        tables['summary']['population'] = population
        tables['summary']['entity_filter'] = str(entity) if entity is not None else '(all selected entities)'
        tables['summary']['entities_in_population'] = rows['__entity'].nunique()
        return tables

    def filter(self, column, *, minimum=None, maximum=None, values=None, include_missing=False,
               minimum_inclusive=True, maximum_inclusive=True, match='exact', case_sensitive=True):
        # VALIDATION LOGIC: Filters are explicit conditions, never evaluated as Python expressions.
        if column not in self.data:
            raise ValueError(f'Unknown filter column: {column!r}')
        if values is not None and (minimum is not None or maximum is not None):
            raise ValueError('Use either category values or numerical boundaries in one filter.')
        from .filtering import category_mask
        if match not in {'exact','contains','starts_with','ends_with'}:
            raise ValueError('match must be exact, contains, starts_with or ends_with.')
        if isinstance(values, str):
            values = [values]
        # CORE LOGIC: STEP 1 — Filter the original population, then recalculate paired coverage and losses.
        # Input: measure=[5,10,20,None], minimum=10, include_missing=False.
        # Output: retained measure=[10,20]; total and paired counts are recalculated on 2 rows.
        # Explanation: Inclusive lower bounds retain the exact declared threshold; missing measurements are excluded explicitly.
        # Trick: Chained filters form intersections; prediction values are reused and no models are fitted.
        source = self.data[column]
        number = pd.to_numeric(source, errors='coerce').replace([np.inf,-np.inf], np.nan)
        mask = source.notna() if values is None else category_mask(source, values, match=match, case_sensitive=case_sensitive)
        if minimum is not None:
            mask &= number.ge(minimum) if minimum_inclusive else number.gt(minimum)
        if maximum is not None:
            mask &= number.le(maximum) if maximum_inclusive else number.lt(maximum)
        mask |= source.isna() & include_missing
        result = compare_predictions(self.data.loc[mask], **self.config)
        # REPORTING LOGIC: Preserve population conditions in every exported review and chained filter.
        condition = dict(column=column,minimum=minimum,maximum=maximum,values=values,include_missing=include_missing,
                         minimum_inclusive=minimum_inclusive,maximum_inclusive=maximum_inclusive,
                         match=match,case_sensitive=case_sensitive)
        result.filter_history = self.filter_history+[condition]
        result.models = dict(self.models)
        result.prediction_columns, result.prediction_offsets = dict(self.prediction_columns), dict(self.prediction_offsets)
        return result

    def cases(self, limit=20, order='harm'):
        # VALIDATION LOGIC: Case ranking is explanatory, not a cleaning or model-selection rule.
        if limit < 1 or order not in {'harm','help','absolute'}:
            raise ValueError('Use a positive limit and order=harm, help or absolute.')
        # CORE LOGIC: STEP 1 — Rank actual paired records by their loss change or candidate error.
        # Input: ids=[1,2,3], ref_abs=[2,4,1], cand_abs=[1,8,1], order='harm', limit=2.
        # Output: ids=[2,3], loss_change=[4,0].
        # Explanation: Record 2 adds four units of error, record 3 ties, and record 1 improves.
        # Trick: A stable sort preserves source order among ties; these examples do not estimate prevalence.
        rows = self.rows.assign(loss_change=self.rows['__ae_candidate']-self.rows['__ae_reference'])
        key = '__ae_candidate' if order == 'absolute' else 'loss_change'
        return rows.sort_values(key, ascending=order=='help', kind='stable').head(limit)

    def missingness(self):
        # CORE LOGIC: STEP 1 — Count missing metadata/features in the complete supplied evaluation population.
        # Input: segment=['A',None], feature=[1,2].
        # Output: segment missing_n=1, missing_pct=50; feature missing_n=0, missing_pct=0.
        # Explanation: Coverage diagnosis includes rows excluded from the paired error comparison.
        # Trick: This table measures pandas missing values; invalid target/prediction strings are separately audited.
        counts = self.data.isna().sum()
        return pd.DataFrame({'column':counts.index,'missing_n':counts.values,'missing_pct':counts.values/max(len(self.data),1)*100})

    def export(self, folder, slices=None, interactions=None, min_count=30, metric='mae_delta', *,
               inference=None, include_candidate=True, candidate_population='candidate', candidate_top_n=20,
               temporal=None, temporal_population=None, temporal_entity=None, trades=None):
        # FILE IO LOGIC: Export a new immutable review directory; never overwrite model artifacts.
        from .report import export_comparison
        return export_comparison(self, folder, slices, interactions, min_count, metric,
                                 inference=inference, include_candidate=include_candidate,
                                 candidate_population=candidate_population, candidate_top_n=candidate_top_n,
                                 temporal=temporal, temporal_population=temporal_population, temporal_entity=temporal_entity,
                                 trades=trades)


def compare_predictions(data, actual, reference, candidate, *, reference_name=None, candidate_name=None,
                        id_column=None, time_column=None, entity_column=None, error_scale=1.0, unit='units',
                        timezone='UTC', tolerance=1.0, reference_offset=None, candidate_offset=None):
    """Compare two prediction columns on a common sample; no model training or automatic split."""
    # VALIDATION LOGIC: Empty comparisons are represented explicitly; invalid configuration is an error.
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be a nonnegative finite value in the displayed error unit.')
    frame = read_data(data, string_columns=[id_column, entity_column])
    reference_name, candidate_name = reference_name or reference, candidate_name or candidate
    if reference_name == candidate_name:
        raise ValueError('Use two distinct model display names.')
    config = dict(actual=actual, reference=reference, candidate=candidate, reference_name=reference_name,
                  candidate_name=candidate_name, id_column=id_column, time_column=time_column,
                  entity_column=entity_column, error_scale=error_scale, unit=unit, timezone=timezone,
                  tolerance=tolerance, reference_offset=reference_offset, candidate_offset=candidate_offset)
    # DATA PREPARATION LOGIC: One shared adapter supplies all subsequent plots, slices and exports.
    options = {k:v for k,v in config.items() if k not in {'reference_name','candidate_name','unit','tolerance'}}
    rows, coverage = paired_rows(frame, **options)
    return Comparison(frame, rows, coverage, config)


@dataclass
class Model:
    # CONFIGURATION LOGIC: Each estimator receives its own explicit ordered feature list.
    name: str
    estimator: object
    features: list
    offset: str = None


def _predict(frame, model):
    # VALIDATION LOGIC: Require explicit feature names and a regression-style one-value-per-row result.
    if not model.features or len(set(model.features)) != len(model.features):
        raise ValueError('Each model needs a nonempty, unique, ordered feature list.')
    missing = set(model.features)-set(frame)
    if missing:
        raise ValueError(f'{model.name} is missing features: {sorted(missing)}')
    # INFERENCE LOGIC: Invoke predict only; never fit, tune, reorder columns or impute model inputs.
    prediction = model.estimator.predict(frame.loc[:,model.features])
    if isinstance(prediction, (pd.Series,pd.DataFrame)) and not prediction.index.equals(frame.index):
        raise ValueError(f'{model.name} returned a pandas result with a different row index.')
    values = np.asarray(prediction)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:,0]
    if values.ndim != 1 or len(values) != len(frame):
        raise ValueError(f'{model.name}.predict must return one regression prediction per input row.')
    return values


def compare_models(data, actual, reference, candidate, **kwargs):
    """Run predict on two already-fitted Model objects with different feature sets if needed."""
    # VALIDATION LOGIC: Prediction metadata must not conflict with caller-supplied comparison arguments.
    frame = read_data(data, string_columns=[kwargs.get('id_column'), kwargs.get('entity_column')])
    columns = ['Reference prediction','Candidate prediction']
    if set(columns) & set(frame):
        raise ValueError('Rename existing Reference prediction/Candidate prediction columns first.')
    if not isinstance(reference, Model) or not isinstance(candidate, Model):
        raise TypeError('Wrap both fitted estimators in Model(name, estimator, features, offset=None).')
    # INFERENCE LOGIC: Prediction only, using explicit independent feature schemas.
    frame[columns[0]], frame[columns[1]] = _predict(frame, reference), _predict(frame, candidate)
    result = compare_predictions(frame, actual, *columns, reference_name=reference.name,
                               candidate_name=candidate.name, reference_offset=reference.offset,
                               candidate_offset=candidate.offset, **kwargs)
    # OBJECT LIFECYCLE LOGIC: Keep fitted estimators in memory for optional explanations; never serialize them.
    result.models = {m.name:Model(m.name,m.estimator,list(m.features),m.offset) for m in [reference,candidate]}
    return result
