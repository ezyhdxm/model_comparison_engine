"""An explicit-Apply notebook workbench for two-model prediction diagnostics."""
# SETUP LOGIC: Widgets and figures are created on demand; importing this module reads no data.
from html import escape
from io import BytesIO
from dataclasses import replace
import re
import numpy as np
from pandas.api.types import is_numeric_dtype
import ipywidgets as w
from IPython.display import display
from .data import read_data, candidate_rows as prepare_candidate_rows
from .ui_style import control, row, section, disclosure, hero, badge, table_html, style_root
from .engine import Comparison, compare_predictions
from .inference import InferenceConfig
from .temporal import TemporalConfig
from .temporal_interpretation import temporal_interpretation
from . import temporal_plots
from .diagnostics import candidate_tables
from .slices import Slice, default_slices
from . import plots, inference_plots, diagnostic_plots


def _trade_configuration(settings=None, **changes):
    # CONFIGURATION LOGIC: Resolve the live module instead of a class alias cached before a notebook update.
    from . import trade_view
    if not hasattr(trade_view, 'normalize_trade_config'):
        raise RuntimeError(f'The trade-view module is outdated or still cached at {trade_view.__file__}. '
                           'Update the complete package, then reload trade_view, trade_plots, report, ui and '
                           'model_comparison_engine in that order and recreate the panel. No kernel restart is needed.')
    return trade_view.normalize_trade_config(settings, **changes)


def _edges(text):
    # CONFIGURATION LOGIC: Parse literal numerical boundaries only; no eval or executable expressions.
    if not text.strip():
        return None
    try:
        return [float(value.strip()) for value in text.split(',')]
    except ValueError as exc:
        raise ValueError('Bin edges must be comma-separated numbers, e.g. -inf,0,1,inf.') from exc


def _image(figure):
    # PLOTTING LOGIC: Render one complete PNG, avoiding the duplicate pyplot notebook display path.
    buffer = BytesIO()
    figure.savefig(buffer,format='png',dpi=135)
    return w.Image(value=buffer.getvalue(),format='png',layout=w.Layout(width='100%'))


def _preview(table):
    # UI LOGIC: Bound the on-screen table; complete tables remain in the export and public result API.
    columns = [c for c in ['group','x','y','n','reference_mae','candidate_mae','mae_improvement_pct','p95_delta','low_support'] if c in table]
    shown = table.loc[:,columns]
    return w.HTML(table_html(shown)+f'<p class="analysis-help">{len(table):,} total groups; export includes every group and metric.</p>')


def _table(table, title):
    # UI LOGIC: Keep full precision in public tables while bounding the notebook's visible records.
    return w.HTML(table_html(table, title=title))


def _guidance(table):
    # UI LOGIC: Narrative diagnostic findings wrap as cards instead of extending a numeric table horizontally.
    cards = []
    for item in table.to_dict('records'):
        title = escape(str(item.get('topic','Interpretation')))
        body = ''.join('<p><b>'+label+':</b> '+escape(str(item.get(key,'')))+'</p>' for key,label in
                       [('observation','Observed'),('interpretation','Meaning'),('next_check','Next check')])
        cards.append('<article class="analysis-guidance-card"><h4>'+title+'</h4>'+body+'</article>')
    return w.HTML('<div class="analysis-guidance">'+''.join(cards)+'</div>')


def _trade_summary(tables):
    # UI LOGIC: Present already-computed coverage as wrapping cards rather than a single wide metadata row.
    summary, settings = tables['summary'].iloc[0], tables['settings']
    count = lambda key: f'{int(summary[key]):,}'
    color = settings['side_column'] or settings['dealer_column'] or settings['counterparty_column'] or 'No category column supplied'
    quantity = settings['quantity_column'] or 'Not supplied: uniform minimum marker area'
    color_note = ('Supplied side codes D, B and S use purple, blue and orange; their meaning is not inferred.' if settings['side_column'] else
                  'No side column is mapped; dealer, then counterparty, supplies fallback category colors when available.')
    view = settings.get('point_view','residual')
    view_labels = {'residual':'Prediction minus actual · '+str(summary['error_unit']),
                   'within_entity':'Deviation from the same entity mean actual · '+str(summary['error_unit']),
                   'level':'Original target level · source target units'}
    marks = ('Residuals use x markers; the zero line means an exact prediction.' if view == 'residual' else
             'Actual values use hollow circles; predictions use x markers.')
    focus = 'All applied paired records' if settings.get('focus_entity') is None else 'Exact entity: '+str(settings['focus_entity'])
    sampled = f"{int(summary.get('sampled_rows',summary['plotted_rows'])):,}"
    cards = [
        ('Evaluation records', ['Actual column: '+str(summary.get('actual_column','the configured Actual column'))+'. '+focus+'.',
            count('paired_rows')+' paired records in this trade view; '+f"{int(summary.get('input_paired_rows',summary['paired_rows'])):,}"+' paired records in the applied comparison.',
            count('timed_rows')+' have timestamps; '+count('missing_time_rows')+' have no usable timestamp. Untimed records can still appear in prediction-versus-actual plots.']),
        ('Displayed points', [view_labels[view]+'. '+sampled+' sampled records; '+count('plotted_rows')+' have drawable coordinates.',
            ('Seeded uniform sample, seed '+str(summary['random_state'])+'.' if summary['sampled'] else 'Every paired record is included in the point sample.')+
            ' Sampling changes the point chart only; interval metrics use all timestamped paired records in this focus.',
            str(summary.get('view_note','Residuals expose prediction error without pooling unrelated instruments into a level-fit chart.'))]),
        ('Time-bin coverage', [str(summary['frequency'])+' intervals · '+str(summary['timezone'])+' labels · '+count('local_dates')+
            ' occupied local dates across '+count('calendar_span_days')+' calendar days.',
            count('supported_bins')+' of '+count('total_bins')+' bins meet minimum N='+count('min_count')+'; '+count('empty_bins')+' bins are empty.',
            'Nights, weekends and other empty intervals remain gaps, not zero errors. Record-count bars show the observations behind each interval.']),
        ('Point encodings', ['Color: '+str(color)+'. '+color_note,
            marks+' Separate panels show each model. Quantity: '+str(quantity)+'. Dealer and counterparty remain exact hover metadata.']),
    ]
    body = ''.join('<article class="analysis-guidance-card"><h4>'+escape(title)+'</h4>'+''.join('<p>'+escape(text)+'</p>' for text in paragraphs)+'</article>' for title, paragraphs in cards)
    return w.HTML('<div class="analysis-guidance">'+body+'</div>')


def _filter_text(condition):
    # UI LOGIC: Present the recorded population in readable terms instead of a Python configuration dict.
    column, parts = condition['column'], []
    if condition['minimum'] is not None:
        parts.append(f'{column} {">=" if condition["minimum_inclusive"] else ">"} {condition["minimum"]:,.8g}')
    if condition['maximum'] is not None:
        parts.append(f'{column} {"<=" if condition["maximum_inclusive"] else "<"} {condition["maximum"]:,.8g}')
    if condition['values'] is not None:
        mode = {'exact':'equals','contains':'contains','starts_with':'starts with','ends_with':'ends with'}[condition.get('match','exact')]
        parts.append(f'{column} {mode} '+ ' or '.join(repr(str(v)) for v in condition['values']))
        if not condition.get('case_sensitive',True):
            parts.append('ignoring case')
    text = ' and '.join(parts) or f'{column} is present'
    return text+(' (including missing)' if condition['include_missing'] else '')


def _entity_columns(columns):
    # UI LOGIC: Rank identifier-like column names without inspecting records or changing the selected mapping.
    priorities = {'entityid':0,'cusip':0,'cusipid':0,'isin':0,'bondid':0,'securityid':0,
                  'instrumentid':0,'securityidentifier':0,'bond':1,'entity':1,'issuerid':2,'issuer':2,'ticker':3}
    def score(column):
        normalized = re.sub(r'[^a-z0-9]','',str(column).lower())
        exact = priorities.get(normalized)
        if exact is not None:
            return exact
        return 1 if any(token in normalized for token in ['cusip','isin','bondid','entityid','securityid','instrumentid']) else 9
    return sorted(columns,key=score)


class ComparisonPanel:
    # UI LOGIC: Input edits do not alter the applied comparison or exported results until Apply succeeds.
    def __init__(self, data=None, *, actual=None, predictions=None, id_column=None, time_column=None,
                 entity_column=None, error_scale=1, unit='units', timezone='UTC',
                 default_slices=None, tolerance=1, reference_offset=None, candidate_offset=None,
                 inference=None, candidate_population='candidate', temporal=None, prediction_offsets=None, trades=None):
        # CONFIGURATION LOGIC: Preserve advanced API settings while exposing common temporal choices in the form.
        self.temporal_defaults = TemporalConfig(**temporal) if isinstance(temporal,dict) else temporal
        if self.temporal_defaults is not None and not isinstance(self.temporal_defaults,TemporalConfig):
            raise TypeError('temporal must be a TemporalConfig, a dictionary, or None.')
        self.trade_defaults = trades
        self.result, self.applied_specs, self.busy = None, None, False
        self._applied_control_state = None
        self.supplied_slices = default_slices
        self.prediction_mapping = predictions
        self.models = {}
        self._suggestion_cache = {}
        self.offsets = dict(reference_offset=reference_offset,candidate_offset=candidate_offset)
        self.initial_filters = []
        self.initial_names = None
        if isinstance(data,Comparison):
            existing = data
            actual, id_column = existing.config['actual'], existing.config['id_column']
            time_column, entity_column = existing.config['time_column'], existing.config['entity_column']
            error_scale, unit, timezone = existing.config['error_scale'], existing.unit, existing.config['timezone']
            tolerance = existing.tolerance
            predictions = dict(existing.prediction_columns)
            self.prediction_mapping = predictions
            self.models = dict(existing.models)
            prediction_offsets = dict(existing.prediction_offsets)
            self.offsets = {k:existing.config[k] for k in ['reference_offset','candidate_offset']}
            self.initial_filters = list(existing.filter_history)
            self.initial_names = (existing.reference_name,existing.candidate_name)
            data = existing.data
        self.data = read_data(data,string_columns='all') if data is not None else None
        self.path = w.Text(description='Data path:',placeholder='Local CSV / Parquet path',layout=w.Layout(width='80%'))
        self.load = w.Button(description='Load data',button_style='info')
        self.load.on_click(self.load_data)
        self.actual = w.Dropdown(description='Actual:')
        self.reference, self.candidate = w.Dropdown(description='Reference:'),w.Dropdown(description='Candidate:')
        self.swap = w.Button(description='Swap models',tooltip='Exchange the reference and candidate; click Apply to refresh results.')
        self.swap.on_click(self.swap_models)
        self.model_help = w.HTML()
        self.identity, self.time, self.entity = [w.Dropdown(description=label) for label in ['Record ID:','Timestamp:','Entity ID:']]
        self.scale, self.unit, self.zone = w.FloatText(value=error_scale,description='Error scale:'),w.Text(value=unit,description='Unit:'),w.Text(value=timezone,description='Timezone:')
        self.tolerance = w.FloatText(value=tolerance,description='Tolerance:')
        self.first, self.second = w.Dropdown(description='Slice:'),w.Dropdown(description='Cross with:')
        self.first_bins, self.second_bins = w.Text(description='Slice bins:'),w.Text(description='Cross bins:')
        self.first_right, self.second_right = w.Checkbox(value=True,description='Slice bins (a,b]'),w.Checkbox(value=True,description='Cross bins (a,b]')
        self.minimum = w.BoundedIntText(value=30,min=1,max=100000000,description='Minimum N:')
        self.top_n = w.BoundedIntText(value=20,min=1,max=50,description='Top groups:')
        self.metric = w.Dropdown(options=[(label,key) for key,label in plots.METRICS.items()],description='Metric:',layout=w.Layout(width='470px'))
        # UI LOGIC: Inference choices are pending until Apply snapshots them with the population and slices.
        inference = inference or InferenceConfig()
        if isinstance(inference,dict):
            inference = InferenceConfig(**inference)
        self.loss = w.Dropdown(options=[('Absolute error','absolute'),('Squared error','squared')],value=inference.loss,description='Test loss:')
        self.test_unit = w.Dropdown(options=[('Auto: date if supplied','auto'),('Record','record'),('Date','date'),('Entity','entity')],
                                    value=inference.unit,description='Test unit:')
        self.correction = w.Dropdown(options=[('BY: more conservative','by'),('BH: less conservative','bh'),('None: unadjusted','none')],value=inference.correction,description='Multiple-test correction:')
        self.alpha = w.BoundedFloatText(value=inference.alpha,min=.000001,max=.999999,step=.01,description='Alpha:')
        self.min_units = w.BoundedIntText(value=inference.min_units,min=2,max=100000000,description='Min units:')
        self.candidate_population = w.Dropdown(options=[('All valid candidate records','candidate'),('Paired records only','paired')],
                                              value=candidate_population,description='Candidate diagnostic sample:',layout=w.Layout(width='420px'))
        self.filters = [self.make_filter(number) for number in range(1,5)]
        self.offset_mapping = dict(prediction_offsets or {})
        self.apply = w.Button(description='Apply comparison',button_style='primary')
        self.apply.on_click(self.run)
        self.export_path = w.Text(value='outputs/model_comparisons',description='Export to:',layout=w.Layout(width='70%'))
        self.export_button = w.Button(description='Export applied review',disabled=True)
        self.export_button.on_click(self.export)
        self.status = w.HTML('Load a table or pass a DataFrame. This workbench never fits a model.')
        self.applied = w.HTML()
        self.views = w.Tab(children=[w.HTML('Apply to calculate the common sample.')])
        self.views.set_title(0,'Results')
        self._make_widget()
        self.entity.observe(self._refresh_trade_entities,names='value')
        self.first.observe(lambda _:self.set_bins(self.first,self.first_bins,self.first_right),names='value')
        self.second.observe(lambda _:self.set_bins(self.second,self.second_bins,self.second_right),names='value')
        if self.data is not None:
            self.configure(actual,id_column,time_column,entity_column)
        self._watch_controls()

    def _make_widget(self):
        # UI LOGIC: Present essential choices first and keep specialist settings in compact disclosures.
        # UI LOGIC: Time diagnostics are opt-in and become part of the next explicit Apply.
        settings = self.temporal_defaults or TemporalConfig()
        frequencies = [('15 minutes','15min'),('Hourly','1h'),('Daily','1D'),('Weekly','7D')]
        if settings.frequency not in dict(frequencies).values():
            frequencies.append((settings.frequency,settings.frequency))
        self.temporal_enabled = w.Checkbox(value=self.temporal_defaults is not None, description='Include time-series diagnostics')
        self.temporal_frequency = w.Dropdown(options=frequencies,value=settings.frequency,description='Time interval:')
        self.temporal_signal = w.Dropdown(options=[('Mean residual (bias)','bias'),('Mean absolute error','mae')],
                                           value=settings.signal,description='Time-series signal:')
        self.temporal_rolling = w.BoundedIntText(value=settings.rolling_bins,min=1,max=10000,description='Rolling window (bins):')
        self.temporal_min_count = w.BoundedIntText(value=settings.min_bin_count,min=1,max=100000000,description='Minimum records per bin:')
        self.temporal_max_lag = w.BoundedIntText(value=settings.max_lag,min=0,max=1000,description='Autocorrelation lags:')
        self._make_trade_controls()
        self.pending = w.HTML(badge('Ready to configure'))
        data_inputs = section('Data and predictions',row(self.path,self.load),row(self.actual,self.reference,self.candidate,self.swap),self.model_help,
            w.HTML('<p class="analysis-help"><b>Reference</b> is the model used as your comparison benchmark; <b>Candidate</b> is the model you are investigating. '
                   'Either can be any supplied model. Candidate minus reference error below zero means the candidate is better. '
                   'Switch the pair, then Apply; saved predictions are reused.</p>'),
            disclosure('Identifiers, units and timestamp settings',row(self.identity,self.time,self.entity),
                       row(self.scale,self.unit,self.zone,self.tolerance),
                       w.HTML('<p class="analysis-help"><b>Entity ID</b> identifies the same instrument across repeated observations, such as a bond identifier. '
                              'Likely identifier names appear first; ordering is a suggestion, not a validated mapping. '
                              'Choose issuer instead only when you intentionally want issuer-level event sequences.</p>')),
            note='Provide any number of prediction columns, then select two models for the applied comparison.',step='01')
        slice_controls = section('Choose the view',row(self.first,self.second,self.metric),row(self.minimum,self.top_n),
            disclosure('Custom bin boundaries',row(self.first_bins,self.first_right),row(self.second_bins,self.second_right),
                       w.HTML('<p class="analysis-help">Comma-separated edges, for example -inf, 0, 1, inf. Leave blank for unbinned groups.</p>')),
            note='Negative candidate-minus-reference error deltas indicate improvement. Low-support groups remain visibly flagged.',step='02')
        inference_controls = disclosure('Inference and candidate diagnostics',
            row(self.loss,self.test_unit,self.correction),row(self.alpha,self.min_units,self.candidate_population),
            w.HTML('<div class="analysis-help"><p><b>Why correction?</b> Testing many slices increases the chance of false discoveries. '
                   'Correction adjusts p-values into q-values within each displayed table. '
                   '<b>BY</b> controls false discovery rate under arbitrary dependence among tests and is the conservative default. '
                   '<b>BH</b> is less conservative and relies on independence or certain positive dependence conditions. '
                   '<b>None</b> uses raw p-values, suitable for exploratory reading without a multiple-testing guarantee.</p>'
                   '<p><b>Test unit</b> controls the observations used by the paired t-test. Date/entity means receive equal weight; '
                   'MAE tables still weight individual records equally. These settings do not remove dependence between dates or entities. '
                   'Negative t favors the candidate. Minimum records and units must both pass. '
                   '<b>Alpha</b> is the decision threshold; q ≤ alpha is flagged significant, not necessarily economically meaningful.</p>'
                   '<p><b>Candidate diagnostic sample</b> applies only to the candidate\'s own residual diagnostics. '
                   'The two-model comparison always uses paired records: finite actual and usable predictions from both models.</p></div>'))
        temporal_controls = disclosure('Optional time-series diagnostics',row(self.temporal_enabled),
            row(self.temporal_frequency,self.temporal_signal),row(self.temporal_rolling,self.temporal_min_count,self.temporal_max_lag),
            w.HTML('<p class="analysis-help">Enable before Apply to inspect trends, rolling errors, autocorrelation, spectral peaks and changes. '
                   'Requires a timestamp column. These diagnostics describe the selected records.</p>'))
        trade_controls = disclosure('Trade-level and intraday views',row(self.trade_enabled,self.trade_frequency,self.trade_max_points),
            row(self.trade_point_view,self.trade_entity),self.trade_entity_help,
            row(self.trade_side,self.trade_counterparty,self.trade_dealer,self.trade_quantity),
            w.HTML('<p class="analysis-help"><b>Residual</b> is the default: prediction minus actual in the configured error unit, with zero meaning an exact prediction. '
                   '<b>Within entity</b> subtracts the same entity\'s mean actual value from both actual and predictions, using all focused paired records before sampling, then applies Error scale. '
                   'This retrospective centering is a diagnostic, not a model feature. <b>Level</b> uses original target units; pooled bonds with very different levels can create a misleading impression of fit. '
                   'Focus entity affects only Trade points, its interval summaries and their exports; other tabs keep the applied population. '
                   'Map optional transaction-side, counterparty type, dealer ID and quantity columns explicitly; these describe records and are not inferred from model names. '
                   'The side column controls color: supplied D/B/S codes use purple/blue/orange, with no automatic interpretation of their meaning. '
                   'Residuals use x markers and a zero-error line. Within-entity and level views use hollow circles for actuals and x markers for predictions, with separate panels per model. '
                   'Quantity controls marker size, while dealer and counterparty remain available in hover. '
                   'Auto interval uses 30 minutes for a span of up to 3 local days, 1 hour for up to 14, otherwise daily. '
                   'The point chart uses a reproducible bounded sample; interval metrics use all paired records.</p>'))
        filters = disclosure('Optional population filters',*[item['widget'] for item in self.filters],
            w.HTML('<p class="analysis-help">Active filter rows are combined with AND. Values separated by semicolons use OR within one row. '
                   'Use Contains for a fragment such as ALPHA; matching treats text literally, not as a regular expression. '
                   'Suggestions use at most the first 10,000 nonmissing rows, so type any value even if absent from the list. '
                   'Blank conditions leave the population unrestricted. Filters only change evaluation records, never the stored predictions.</p>'))
        actions = row(self.apply,self.pending).add_class('analysis-actions')
        self.status.add_class('analysis-status')
        self.applied.add_class('analysis-applied')
        self.views.layout.width = '100%'
        self.views.children = [w.HTML('<div class="analysis-card-heading"><h3>Your review will appear here</h3>'
            '<p>Choose your inputs and click Apply comparison. Results and exports retain the last successful applied configuration.</p></div>')]
        self.views.set_title(0,'Results')
        export = section('Save the applied review',row(self.export_path,self.export_button),
                         note='Export figures, full tables and an HTML report using the last successful comparison.')
        self.widget = style_root(w.VBox([hero('Compare two models','Inspect a reference and candidate on the same records, then explore where errors improve or deteriorate.',
            tags=('Multiple model choices','Explicit Apply','Portable review')),data_inputs,slice_controls,inference_controls,temporal_controls,trade_controls,filters,
            actions,self.status,self.applied,self.views,export]))
        for item in [self.path,self.export_path]:
            control(item,wide=True)

    def _watch_controls(self):
        # UI LOGIC: Edits only update a badge; expensive work remains behind the explicit Apply button.
        names = ['actual','reference','candidate','identity','time','entity','scale','unit','zone','tolerance',
                 'first','second','first_bins','second_bins','first_right','second_right','minimum','top_n','metric']
        names += ['loss','test_unit','correction','alpha','min_units','candidate_population',
                  'temporal_enabled','temporal_frequency','temporal_signal','temporal_rolling','temporal_min_count','temporal_max_lag']
        names += ['trade_enabled','trade_frequency','trade_max_points','trade_side','trade_counterparty','trade_dealer','trade_quantity',
                  'trade_point_view','trade_entity']
        self._pending_controls = [getattr(self,name) for name in names]
        self._pending_controls += [item[key] for item in self.filters for key in ['column','low','high','strict','categories','match','case_sensitive']]
        for item in self._pending_controls:
            control(item,wide=item is self.metric)
            item.observe(self._pending_changed,names='value')
        self._pending_changed()

    def _make_trade_controls(self):
        # UI LOGIC: Optional role mappings describe transaction records independently of prediction features.
        self._trade_config = _trade_configuration(self.trade_defaults)
        frequencies = [('Auto: based on date span','auto'),('15 minutes','15min'),('30 minutes','30min'),('Hourly','1h'),('Daily','1D')]
        if self._trade_config.frequency not in dict(frequencies).values():
            frequencies.append((self._trade_config.frequency,self._trade_config.frequency))
        self.trade_enabled = w.Checkbox(value=True,description='Include trade-level views')
        self.trade_frequency = w.Dropdown(description='Comparison interval:',options=frequencies,value=self._trade_config.frequency)
        self.trade_max_points = w.BoundedIntText(description='Maximum plotted trades:',value=self._trade_config.max_points,min=1,max=100000)
        self.trade_point_view = w.Dropdown(description='Point view:',options=[('Residual: prediction minus actual','residual'),
            ('Within entity: common actual mean','within_entity'),('Level: original target units','level')],
            value=getattr(self._trade_config,'point_view','residual'))
        self.trade_entity = w.Dropdown(description='Focus entity (trade tab only):',options=[('All applied paired records',None)])
        self.trade_entity_help = w.HTML()
        self._trade_entity_counts_result,self._trade_entity_counts_data_id = None,None
        self.trade_side,self.trade_counterparty,self.trade_dealer,self.trade_quantity = [w.Dropdown(description=label) for label in
            ['Side / color column:','Counterparty type column:','Dealer ID column:','Quantity / size column:']]

    def _refresh_trade_entities(self, change=None, *, comparison=None, initial=False):
        # UI LOGIC: Entity options keep exact source values; string labels are presentation only.
        column = self.entity.value
        desired = getattr(self._trade_config,'focus_entity',None) if initial else self.trade_entity.value
        if change is not None:
            desired = None
        if self.data is None or column is None or column not in self.data:
            self.trade_entity.options = [('All applied paired records',None)]
            self.trade_entity_help.value = '<p class="analysis-help">Choose Entity ID above to focus this trade view on one exact instrument or issuer.</p>'
            return
        if comparison is not None:
            self._trade_entity_counts_result,self._trade_entity_counts_data_id = comparison,id(self.data)
        applied = self._trade_entity_counts_result
        available = applied is not None and self._trade_entity_counts_data_id == id(self.data) and applied.config['entity_column'] == column
        # CORE LOGIC: STEP 1 — Order all known entities by their usable paired counts when a comparison exists.
        # Input: supplied entities=['A','A','B','B','C'], applied paired entities=['B','A','B'].
        # Output: selector counts in order={'B':2,'A':1,'C':0}; no supplied entity is truncated.
        # Explanation: Before the first Apply, supplied counts are shown instead and explicitly labeled as such.
        # Trick: Reindex retains entities with zero paired records; option values preserve their original data types.
        supplied = self.data[column].value_counts(dropna=True)
        supplied = supplied.loc[supplied.gt(0)]
        counts = applied.rows[column].value_counts(dropna=True).reindex(supplied.index,fill_value=0) if available else supplied
        counts = counts.sort_values(ascending=False,kind='stable')
        # UI LOGIC: Changing the display order must not change an already selected exact-value focus.
        scope = 'paired' if available else 'supplied'
        options = [('All applied paired records',None)]+[(f'{value} · {int(number):,} {scope} records',value) for value,number in counts.items()]
        if desired is not None and desired not in counts.index:
            options.append((f'{desired} · absent from loaded data',desired))
        self.trade_entity.options = options
        self.trade_entity.value = desired
        self.trade_entity_help.value = ('<p class="analysis-help">All '+str(len(counts))+' nonmissing values of <b>'+escape(str(column))+
            '</b> are selectable; values are matched exactly without converting numeric IDs to text. Counts use '+
            ('the last successfully applied paired population.' if available else 'the supplied data until the first Apply.')+
            ' Missing entity IDs remain in the all-records view. Changes take effect on Apply.</p>')

    def _control_state(self):
        # UI LOGIC: Compare literal widget values plus the loaded-table identity; export-path edits are independent.
        return (id(self.data),tuple(item.value for item in self._pending_controls))

    def _pending_changed(self, change=None):
        # UI LOGIC: Reverting every edit restores the applied badge without recalculating anything.
        if self.busy:
            self.pending.value = badge('Working · controls are locked','busy')
        elif self._applied_control_state is None:
            self.pending.value = badge('Not applied · choose inputs and Apply','neutral')
        elif self._control_state() != self._applied_control_state:
            self.pending.value = badge('Pending changes · Apply to update results','pending')
        else:
            self.pending.value = badge('Applied · results match these controls','ready')

    def _set_busy(self, busy, *, exporting=False):
        # UI LOGIC: Lock mutable inputs while an explicit action runs and restore the existing result afterward.
        self.busy = busy
        self.apply.disabled = self.load.disabled = busy
        self.swap.disabled = busy
        self.export_button.disabled = busy or self.result is None
        for item in self._pending_controls:
            item.disabled = busy
        self.apply.description = 'Comparing…' if busy and not exporting else 'Apply comparison'
        self.export_button.description = 'Exporting…' if busy and exporting else 'Export applied review'
        self._pending_changed()

    def swap_models(self, _=None):
        # UI LOGIC: Swap only pending model choices; the applied comparison and export remain unchanged.
        if not self.busy:
            reference, candidate = self.reference.value, self.candidate.value
            self.reference.value, self.candidate.value = candidate, reference

    def make_filter(self, number):
        # UI LOGIC: Searchable suggestions support exact or partial literal text without requiring full issuer names.
        fields = dict(column=w.Dropdown(description='Column:'),low=w.Text(description='Lower:'),high=w.Text(description='Upper:'),
                      strict=w.Checkbox(value=False,description='Strict lower >'),
                      categories=w.Combobox(description='Values / fragments:',placeholder='Type or select; separate alternatives with ;',ensure_option=False),
                      match=w.Dropdown(description='Text match:',options=[('Exact value','exact'),('Contains text','contains'),('Starts with','starts_with'),('Ends with','ends_with')]),
                      case_sensitive=w.Checkbox(value=False,description='Case sensitive'))
        fields['column'].observe(lambda _:self.filter_suggestions(fields),names='value')
        fields['widget'] = disclosure(f'Filter {number}',row(fields['column'],fields['low'],fields['high']),
                                      row(fields['categories'],fields['match'],fields['case_sensitive'],fields['strict']),opened=number==1)
        return fields

    def filter_suggestions(self, fields):
        # UI LOGIC: A bounded, cached suggestion list assists typing but never restricts accepted filter values.
        column = fields['column'].value
        if column is None or self.data is None:
            fields['categories'].options = ()
            return
        if column not in self._suggestion_cache:
            sample = self.data[column].dropna().iloc[:10000].astype(str)
            self._suggestion_cache[column] = tuple(sample.value_counts().head(250).index)
        fields['categories'].options = self._suggestion_cache[column]

    def configure(self, actual=None, identity=None, time=None, entity=None):
        # UI LOGIC: Populate selectors without guessing hidden metadata or running a prediction.
        columns = list(self.data.columns)
        if not columns:
            raise ValueError('The input table has no columns.')
        self.actual.options = columns
        self.actual.value = actual if actual in columns else next((c for c in ['actual','target'] if c in columns),columns[0])
        mapping = self.prediction_mapping or {str(c):c for c in columns if c != self.actual.value}
        if len(mapping) < 2:
            raise ValueError('Supply at least two prediction columns, or use compare_models first.')
        if not set(mapping.values()).issubset(columns):
            raise ValueError('A configured prediction column does not exist in this data.')
        if not set(self.offset_mapping).issubset(mapping):
            raise ValueError('prediction_offsets contains a model name absent from predictions.')
        if any(column is not None and column not in columns for column in self.offset_mapping.values()):
            raise ValueError('A prediction offset column does not exist in this data.')
        self.mapping = mapping
        self.reference.options = self.candidate.options = list(mapping)
        preferred_reference = self.initial_names[0] if self.initial_names else None
        self.reference.value = preferred_reference or next((c for c in ['Reference','reference_prediction','Reference prediction'] if c in mapping),list(mapping)[0])
        others = [c for c in mapping if c != self.reference.value]
        preferred_candidate = self.initial_names[1] if self.initial_names else None
        self.candidate.value = preferred_candidate or next((c for c in ['Candidate','candidate_prediction','Candidate prediction'] if c in others),others[0])
        self.offset_mapping.setdefault(self.reference.value,self.offsets['reference_offset'])
        self.offset_mapping.setdefault(self.candidate.value,self.offsets['candidate_offset'])
        for control,value,aliases in [(self.identity,identity,['row_id','record_id']),(self.time,time,['time','timestamp']),
                                      (self.entity,entity,['entity_id'])]:
            ranked = _entity_columns(columns) if control is self.entity else columns
            control.options = [('Not supplied',None)]+[(str(c),c) for c in ranked]
            control.value = value if value in columns else next((c for c in aliases if c in columns),None)
        supplied = self.supplied_slices
        defaults = supplied if supplied is not None else (default_slices(columns) if self.time.value else [])
        self.defaults = {s.column:s for s in defaults}
        slice_columns = list(dict.fromkeys(columns+(['__hour','__date'] if self.time.value else [])))
        self.first.options = [(str(c),c) for c in slice_columns]
        self.second.options = [('None',None)]+[(str(c),c) for c in slice_columns]
        predicted = set(mapping.values()) if self.prediction_mapping else {mapping[self.reference.value],mapping[self.candidate.value]}
        excluded = predicted | {self.actual.value,self.identity.value,self.time.value,self.entity.value}
        metadata = [c for c in columns if c not in excluded]
        categorical = [c for c in metadata if not is_numeric_dtype(self.data[c])]
        preferred = next(iter(categorical or metadata),columns[0])
        self.first.value = next(iter(self.defaults),preferred)
        self.second.value = None
        self.set_bins(self.first,self.first_bins,self.first_right)
        for item in self.filters:
            item['column'].options = [('No filter',None)]+[(str(c),c) for c in columns]
        for selector, name in [(self.trade_side,'side_column'),(self.trade_counterparty,'counterparty_column'),
                               (self.trade_dealer,'dealer_column'),(self.trade_quantity,'quantity_column')]:
            value = getattr(self._trade_config,name)
            selector.options = [('Not supplied',None)]+[(str(c),c) for c in columns]
            selector.value = value if value in columns else None
        self._refresh_trade_entities(initial=True)
        self.model_help.value = ('<p class="analysis-help"><b>'+str(len(mapping))+' selectable prediction series:</b> '+
                                 ', '.join(escape(str(name)) for name in mapping)+'. '+
                                 ('Fitted model objects are attached; feature explanations are available.' if self.models else
                                  'Prediction-only input: residual and slice diagnostics are available; feature explanations require fitted models.')+'</p>')
        self.status.value = f'Loaded {len(self.data):,} rows and {len(columns)} columns. Select predictions and Apply.'

    def set_bins(self, selector, text, right):
        # UI LOGIC: Use explicitly supplied boundaries or generic time intervals; other columns accept custom edges.
        spec = getattr(self,'defaults',{}).get(selector.value)
        text.value = ','.join(str(v) for v in spec.bins) if spec and spec.bins is not None else ''
        right.value = spec.right if spec else True

    def load_data(self, _=None):
        # FILE IO LOGIC: Replace the input only on an explicit load; preserve the applied result on error.
        self._set_busy(True)
        self.load.description = 'Loading…'
        self.status.value = 'Loading the selected table…'
        try:
            data = read_data(self.path.value,string_columns='all')
            self.data = data
            self.initial_filters = []
            self.initial_names = None
            self.models,self._suggestion_cache = {},{}
            self.prediction_mapping,self.offset_mapping = None,{}
            self.offsets = dict(reference_offset=None,candidate_offset=None)
            self.configure()
        except Exception as exc:
            self.status.value = '<b>Input error:</b> '+escape(str(exc))
        finally:
            self.load.description = 'Load data'
            self._set_busy(False)

    def specs(self):
        # CONFIGURATION LOGIC: Snapshot the currently requested boundaries, names and closure.
        first = self.make_spec(self.first,self.first_bins,self.first_right,self.top_n.value)
        second = self.make_spec(self.second,self.second_bins,self.second_right,min(self.top_n.value,20)) if self.second.value else None
        return first,second

    def make_spec(self, selector, text, right, top_n):
        # CONFIGURATION LOGIC: Friendly default labels remain valid only while their boundaries and closure match.
        bins = _edges(text.value)
        original = self.defaults.get(selector.value)
        same_bins = original is not None and ((bins is None and original.bins is None) or
                    (bins is not None and original.bins is not None and np.array_equal(bins,original.bins)))
        labels = original.labels if same_bins and right.value == original.right else None
        return Slice(selector.value,bins,labels,right.value,top_n)

    def paired_views(self, result, first, second, inference, min_count, metric, title):
        # UI LOGIC: A missing reference must not hide the candidate's independently valid records.
        if result.rows.empty:
            return [w.HTML('No paired finite records in the applied population.') for _ in range(3)]
        # REPORTING LOGIC: Each significance table uses the same records and declared slices as its paired metrics.
        overall = result.paired_test(inference=inference,min_count=min_count)
        table = result.cross_slice(first,second,min_count=min_count) if second else result.slice(first,min_count=min_count)
        tested = (result.cross_slice_test(first,second,inference=inference,min_count=min_count) if second else
                  result.slice_test(first,inference=inference,min_count=min_count))
        resolved_unit = result.inference_config(inference).unit
        # PLOTTING LOGIC: Keep the effect-size chart next to statistical evidence and the support needed to interpret it.
        chart = plots.heatmap(table,metric,unit=result.unit,title=title) if second else plots.slice_figure(table,metric,unit=result.unit,title=title)
        significance = inference_plots.significance_heatmap(tested,title=title,unit_label=result.unit)
        loss_unit = result.unit if inference.loss == 'absolute' else f'{result.unit} squared'
        note = w.HTML(f'<p>Two-sided mean {escape(inference.loss)} loss difference ({escape(loss_unit)}); '
                      f'equal weight per {escape(resolved_unit)}. Correction: {inference.correction.upper()}, alpha={inference.alpha:g}. '
                      'Stars show adjusted q thresholds: * ≤ .05, ** ≤ .01, *** ≤ .001; the significant column uses alpha. '
                      '† marks low support. Zero/near-zero variance has no test or stars. '
                      'Intervals are pointwise, not multiplicity-adjusted. No stars does not establish equivalence.</p>')
        slices = w.Tab(children=[w.VBox([_image(chart),_preview(table)]),w.VBox([note,_image(significance),_table(tested,'Paired tests')])])
        slices.set_title(0,'Error metrics')
        slices.set_title(1,'Significance')
        coverage_note = w.HTML('<p><b>Paired</b> means the same records have a finite actual, both reconstructed predictions '
                               'and both scaled errors. Model comparison uses only these common records. '
                               f'Independently evaluable: reference {result.coverage["reference_evaluable"]:,}; '
                               f'candidate {result.coverage["candidate_evaluable"]:,}.</p>')
        return [w.VBox([_image(plots.overview(result)),coverage_note,_table(overall,'Overall paired test')]),slices,
                w.VBox([_image(plots.daily_figure(result.daily(),unit=result.unit)),_table(result.stability(),'Leave-one-date-out sensitivity')])]

    def candidate_view(self, result, first, second, population, min_count, top_n):
        # REPORTING LOGIC: Candidate diagnostics choose their own explicit population and never use reference errors.
        rows = result.candidate_rows(population=population)
        specs = [first,second] if second else [first]
        tables = candidate_tables(rows, specs, min_count=min_count, top_n=top_n,
                                  tolerance=result.tolerance, id_column=result.config['id_column'])
        tables['summary']['population'] = population
        # UI LOGIC: Show population counts before any residual figure to prevent an accidental unequal-sample comparison.
        description = 'all valid candidate records' if population == 'candidate' else 'paired records only'
        note = w.HTML(f'<p><b>{escape(result.candidate_name)}:</b> {len(rows):,} records ({description}); '
                      f'paired comparison: {len(result.rows):,} records; supplied after filters: {len(result.data):,}. '
                      'Candidate and paired metrics are directly comparable only when they use the same records. '
                      'Positive residual means overprediction. Worst cases and slices are descriptive rankings.</p>')
        figure = diagnostic_plots.candidate_figure(rows,tables,name=result.candidate_name,unit=result.unit,error_scale=result.config['error_scale'])
        labels = [('summary','Summary'),('calibration','Calibration'),('residual_quantiles','Residual quantiles'),
                  ('worst_slices','Worst slices'),('worst_cases','Worst cases'),('daily','Dates')]
        detail_views = [_table(tables[key],label) for key,label in labels]
        weakest = diagnostic_plots.worst_slices_figure(tables['worst_slices'],unit=result.unit,title=f'{result.candidate_name}: weakest slices')
        detail_views[3] = w.VBox([_image(weakest),detail_views[3]])
        details = w.Tab(children=detail_views)
        for index,(_,label) in enumerate(labels):
            details.set_title(index,label)
        return w.VBox([note,_image(figure),details])

    def temporal_settings(self):
        # CONFIGURATION LOGIC: Snapshot only on Apply; retain advanced limits supplied through TemporalConfig.
        if not self.temporal_enabled.value:
            return None
        if self.time.value is None:
            raise ValueError('Select a Timestamp column before enabling time-series diagnostics.')
        return replace(self.temporal_defaults or TemporalConfig(),frequency=self.temporal_frequency.value,
                       signal=self.temporal_signal.value,rolling_bins=self.temporal_rolling.value,
                       min_bin_count=self.temporal_min_count.value,max_lag=self.temporal_max_lag.value)

    def temporal_view(self, result, settings, population):
        # UI LOGIC: The default review skips optional temporal computation entirely.
        if settings is None:
            return w.HTML('<p class="analysis-help">Enable Optional time-series diagnostics above, select a timestamp, '
                          'then Apply. Existing predictions are reused; no model is fitted.</p>')
        # REPORTING LOGIC: The temporal population matches the applied candidate diagnostics, including filters.
        tables = result.temporal_diagnostics(settings,population=population)
        note = w.HTML('<p>Fixed elapsed-time bins use UTC. Positive residual means overprediction. '
                      'Missing intervals remain gaps; bins below the minimum support are excluded from time signals. '
                      'Spectral peaks and offline mean-shift candidates are exploratory, with no significance stars. '
                      'Check coverage and population composition before changing a feature or model.</p>')
        figure = temporal_plots.temporal_figure(tables,name=result.candidate_name,unit=result.unit)
        event_figure = temporal_plots.event_lag_figure(tables['event_autocorrelation'],name=result.candidate_name,unit=result.unit)
        labels = [('change_points','Mean shifts'),('autocorrelation','Clock lags'),('event_autocorrelation','Entity event lags'),
                  ('spectrum','Spectrum'),('series','Time bins')]
        details = w.Tab(children=[_table(tables[key],label) for key,label in labels])
        for index,(_,label) in enumerate(labels):
            details.set_title(index,label)
        return w.VBox([note,_guidance(temporal_interpretation(tables,unit=result.unit)),
                       _table(tables['summary'],'Temporal coverage'),_image(figure),_image(event_figure),details])

    def trade_settings(self):
        # CONFIGURATION LOGIC: Record plotting choices with the applied comparison and its exports.
        if not self.trade_enabled.value:
            return None
        return _trade_configuration(self._trade_config,frequency=self.trade_frequency.value,min_count=self.minimum.value,
                       max_points=self.trade_max_points.value,side_column=self.trade_side.value,
                       counterparty_column=self.trade_counterparty.value,dealer_column=self.trade_dealer.value,
                       quantity_column=self.trade_quantity.value,point_view=self.trade_point_view.value,
                       focus_entity=self.trade_entity.value)

    def trade_view(self, result, settings):
        # UI LOGIC: Intraday summaries and sampled point plots reuse predictions and never fit a model.
        if settings is None:
            return w.HTML('<p class="analysis-help">Enable Trade-level views, then Apply. A timestamp is needed only for the time axis and intraday bins. '
                          'Optional role columns make buy/sell, dealer and quantity visible. Filter to one instrument for a readable trade sequence.</p>')
        from . import trade_plots
        tables = result.trade_diagnostics(settings)
        options = dict(reference_name=result.reference_name,candidate_name=result.candidate_name,unit=result.unit)
        button = w.Button(description='Show interactive trade points',button_style='info')
        interactive = w.HTML('<p class="analysis-help">Interactive hover shows trade attributes; click above to load it. The static figures remain available without Plotly.</p>')
        def show_interactive(_):
            # PLOTTING LOGIC: Embed a standalone figure on demand; script execution is isolated in its own frame.
            button.disabled = True
            try:
                figure = trade_plots.trade_interactive(tables,**options)
                document = figure.to_html(full_html=True,include_plotlyjs=True)
                interactive.value = '<iframe title="Interactive trade predictions" sandbox="allow-scripts allow-downloads" style="width:100%;height:1050px;border:0" srcdoc="'+escape(document,quote=True)+'"></iframe>'
            except Exception as exc:
                interactive.value = '<p class="analysis-help">Interactive view unavailable: '+escape(str(exc))+'. Static charts and complete tables remain available.</p>'
            finally:
                button.disabled = False
        button.on_click(show_interactive)
        note = w.HTML('<p class="analysis-help">Error-by-interval metrics use all paired records in this trade view\'s optional entity focus. '
                      'Other comparison tabs retain the globally applied population. Trade points are a reproducible bounded sample. '
                      'The count panel uses bars: empty calendar intervals are gaps, not drops to zero error. No interpolation fills inactive intervals. '
                      'Residual and within-entity views use the configured error unit; original-level views use source target units. '
                      'Within-entity centering uses realized actuals from the focused evaluation sample and is not available as a predictive feature.</p>')
        return w.VBox([note,_trade_summary(tables),
                       _image(trade_plots.intraday_figure(tables,**options)),_image(trade_plots.trade_figure(tables,**options)),
                       row(button),interactive,disclosure('Interval metrics and plotted records',
                       _table(tables['intraday'],'Interval metrics'),_table(tables['points'],'Plotted records'))])

    def explanation_view(self, result):
        # UI LOGIC: Feature explanations are separate explicit actions on the last applied records and fitted models.
        if not result.models:
            return w.HTML('<p class="analysis-help">Feature explanations require fitted model objects and their input columns. '
                          'Pass the result of compare_models() or compare_model_set() to show_comparison(). '
                          'Prediction-only tables support every error diagnostic, but cannot reconstruct model internals or rerun perturbed predictions.</p>')
        preferred = result.candidate_name if result.candidate_name in result.models else next(iter(result.models))
        choices = [(name,name) for name in result.models]
        if result.reference_name in result.models and result.candidate_name in result.models:
            choices.insert(0,('Both applied models',None))
        model = w.Dropdown(description='Explain model:',options=choices,value=preferred)
        population = w.Dropdown(description='Explanation records:',options=[('Paired comparison records','paired'),('All filtered records for this model','model')])
        maximum = w.BoundedIntText(description='Maximum sampled records:',value=500,min=1,max=50000)
        shap_sample = w.Dropdown(description='SHAP record selection:',options=[('Reproducible random sample','random'),('Largest absolute errors','worst')])
        repeats = w.BoundedIntText(description='Permutation repeats:',value=5,min=1,max=100)
        metric = w.Dropdown(description='Permutation metric:',options=[('MAE','mae'),('RMSE','rmse')])
        shuffle = w.Dropdown(description='Shuffle within column:',options=[('All selected records',None)]+[(str(c),c) for c in result.data.columns])
        groups = w.Textarea(description='Feature groups:',placeholder='Quote group = quote_level, quote_age\nActivity group = recent_count, last_trade_age')
        individuals = w.Checkbox(value=True,description='Also test individual features')
        permutation = w.Button(description='Compute permutation importance',button_style='primary')
        shap = w.Button(description='Compute SHAP',button_style='info')
        progress = w.HTML('<p class="analysis-help">Uses the last applied population, even if the comparison controls above have pending changes. No model is retrained.</p>')
        output = w.VBox()
        controls = [model,population,maximum,shap_sample,repeats,metric,shuffle,groups,individuals,permutation,shap]
        for item in controls:
            control(item,wide=item is groups)
        def report_progress(state):
            # UI LOGIC: Report completed prediction batches rather than presenting a silent long-running action.
            progress.value = badge(f'{state.get("stage","Working")} · {state.get("completed",0):,} / {state.get("total",0):,}','busy')
        def calculate(kind):
            # UI LOGIC: Keep expensive explanations behind separate buttons; failed requests retain earlier outputs.
            if self.busy:
                return
            self._set_busy(True)
            for item in controls:
                item.disabled = True
            # CACHEING LOGIC: Publish explanation caches atomically with the rendered panels, including both-model requests.
            previous_explanations = dict(result.explanations)
            try:
                names = [result.reference_name,result.candidate_name] if model.value is None else [model.value]
                if len(names) > 1 and population.value != 'paired':
                    raise ValueError('Choose Paired comparison records when explaining both models so they use the same evaluation population.')
                selected_groups = self.feature_groups(groups.value) if kind == 'permutation' else None
                self.validate_feature_groups(result,names,selected_groups)
                self.status.value = f'Computing {escape(kind)} for the last applied '+escape(', '.join(names))+' records…'
                positions = self.worst_explanation_positions(result,names[-1],population.value,maximum.value) if kind == 'SHAP' and shap_sample.value == 'worst' else None
                panels = []
                for name in names:
                    common = dict(model=name,population=population.value,max_rows=maximum.value,progress=report_progress)
                    if kind == 'permutation':
                        explanation = result.permutation_importance(**common,groups=selected_groups,
                            include_individual=individuals.value,metric=metric.value,repeats=repeats.value,shuffle_within=shuffle.value)
                    else:
                        explanation = result.shap_values(**common,row_positions=positions)
                        explanation.settings['record_selection'] = shap_sample.value
                    panels.append(section(name,*self.explanation_result(explanation,kind),note='Feature contributions describe this fitted model and the applied records.'))
                output.children = panels
                progress.value = badge('Complete · included in the next applied review export','ready')
                self.status.value = 'Feature explanation complete. Results describe the last applied population; no model was fitted.'
            except Exception as exc:
                # CACHEING LOGIC: A failed later model or plot must not replace only part of the applied export state.
                result.explanations.clear()
                result.explanations.update(previous_explanations)
                progress.value = '<b>Explanation error:</b> '+escape(str(exc))
                self.status.value = progress.value+' Earlier explanation results remain available.'
            finally:
                for item in controls:
                    item.disabled = False
                self._set_busy(False)
        permutation.on_click(lambda _:calculate('permutation'))
        shap.on_click(lambda _:calculate('SHAP'))
        help_text = w.HTML('<div class="analysis-help"><p><b>SHAP</b> decomposes model predictions into feature contributions; large contributions do not prove accuracy gains. '
                          'The offset added back to a residual model is reported separately. Random mode summarizes a sample; Largest absolute errors selects from the complete applied population. '
                          'For Both applied models, worst-error selection uses the applied candidate and explains those same rows for both models. '
                          'Attribution differences do not establish accuracy differences.</p>'
                          '<p><b>Permutation</b> measures error increase after shuffling a feature or feature group. Positive increases indicate model reliance. '
                          'Correlated features can substitute for each other: enter a group to shuffle its columns together. '
                          'Shuffling can create unrealistic combinations; use a meaningful within-column restriction if needed. This is not a causal effect or a retraining ablation.</p>'
                          '<p>Use validation data for exploratory feature decisions. Sampled importance is conditional on the applied filters; it does not establish full-population improvement. '
                          'Groups use one line per group: <code>Group name = feature_a, feature_b</code>.</p></div>')
        return w.VBox([help_text,row(model,population,maximum),row(shap_sample),row(repeats,metric,shuffle),
                       disclosure('Grouped permutation settings',row(groups),row(individuals)),row(permutation,shap),progress,output])

    def feature_groups(self, text):
        # CONFIGURATION LOGIC: Parse explicit feature names; this accepts no Python expressions or regular expressions.
        groups = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            name, separator, values = line.partition('=')
            columns = [value.strip() for value in values.split(',') if value.strip()]
            if not separator or not name.strip() or not columns:
                raise ValueError('Each feature group must look like: Quote features = quote_level, quote_age')
            if name.strip() in groups:
                raise ValueError('Feature group names must be unique.')
            groups[name.strip()] = columns
        return groups or None

    def validate_feature_groups(self, result, names, groups):
        # VALIDATION LOGIC: A shared group must exist in every selected model; missing features are never silently discarded.
        for name in names:
            for group, columns in (groups or {}).items():
                missing = set(columns)-set(result.models[name].features)
                if missing:
                    raise ValueError(f'{name}: group {group!r} contains unavailable features {sorted(missing)}. Select one model or edit the group.')

    def worst_explanation_positions(self, result, name, population, limit):
        # CORE LOGIC: STEP 1 — Use the identical positional population accepted by the explanation API.
        # Input: source x=[10,20,30], source index=[7,7,9], paired positions=[0,2], population='paired';
        # name='B', prediction_columns={'B':'prediction'}, prediction_offsets={'B':'offset'}.
        # Output: selected x=[10,30], selected index=[7,9], column='prediction', offset='offset'; local positions are 0 and 1.
        # Explanation: Local record positions refer to the filtered explanation input, not the original table index.
        # Trick: iloc avoids reintroducing an excluded row when source index labels repeat.
        frame = result.data.iloc[result.rows['__source_position']] if population == 'paired' else result.data
        column, offset = result.prediction_columns[name], result.prediction_offsets.get(name)
        # CORE LOGIC: STEP 2 — Rank cached final-level errors without predicting again or including missing targets.
        # Input: actual=[100,100,100], raw prediction=[1,4,-2], offset=[100,100,100], error_scale=1, limit=2.
        # Output: row_positions=[1,2], from absolute final errors [1,4,2].
        # Explanation: Reuse candidate-error preparation to keep finite-error validation and offset reconstruction consistent.
        # Trick: Stable sorting breaks equal-error ties by source order; worst cases are not a representative population sample.
        rows = prepare_candidate_rows(frame,result.config['actual'],column,
            candidate_offset=offset,error_scale=result.config['error_scale'])
        return rows.sort_values('__ae_candidate',ascending=False,kind='stable')['__source_position'].head(limit).tolist()

    def explanation_result(self, explanation, kind):
        # PLOTTING LOGIC: Show global importance plus an explicit local record selector for SHAP contributions.
        from . import explanation_plots
        notes = w.HTML('<p class="analysis-help">'+'<br>'.join(escape(str(note)) for note in explanation.notes)+'</p>')
        tables = disclosure('Complete explanation tables',*[_table(table,name.replace('_',' ').title()) for name,table in explanation.tables.items()])
        if kind == 'permutation':
            return [notes,_image(explanation_plots.plot_permutation(explanation)),tables]
        local = explanation.tables['local_contributions']
        positions = list(dict.fromkeys(local['row_position']))
        selector = w.Dropdown(description='Local record position:',options=positions)
        chart = w.VBox()
        def update_local(_=None):
            # PLOTTING LOGIC: Reuse stored contributions instead of rerunning SHAP when changing the displayed record.
            if selector.value is not None:
                chart.children = [_image(explanation_plots.plot_shap_local(explanation,row_position=selector.value))]
        selector.observe(update_local,names='value')
        update_local()
        return [notes,_image(explanation_plots.plot_shap_global(explanation)),row(control(selector)),chart,tables]

    def run(self, _=None):
        # UI LOGIC: Publish result state only after a successful computation; previous exports remain valid.
        if self.busy:
            return
        self._set_busy(True)
        self.status.value = 'Comparing saved predictions on common rows...'
        try:
            if self.data is None:
                raise ValueError('Load data first.')
            reference,candidate = self.reference.value,self.candidate.value
            if reference == candidate:
                raise ValueError('Choose two different models, or use Swap models to reverse the current pair.')
            result = compare_predictions(self.data,self.actual.value,self.mapping[reference],self.mapping[candidate],
                                         reference_name=reference,candidate_name=candidate,id_column=self.identity.value,
                                         time_column=self.time.value,entity_column=self.entity.value,error_scale=self.scale.value,
                                         unit=self.unit.value,timezone=self.zone.value,tolerance=self.tolerance.value,
                                         reference_offset=self.offset_mapping.get(reference),candidate_offset=self.offset_mapping.get(candidate))
            result.models,result.prediction_columns,result.prediction_offsets = dict(self.models),dict(self.mapping),dict(self.offset_mapping)
            result.filter_history = list(self.initial_filters)
            filters = []
            for item in self.filters:
                has_condition = any(item[k].value.strip() for k in ['low','high','categories'])
                if item['column'].value is not None and has_condition:
                    options = dict(minimum=float(item['low'].value) if item['low'].value.strip() else None,
                                   maximum=float(item['high'].value) if item['high'].value.strip() else None,
                                   values=[v.strip() for v in item['categories'].value.split(';') if v.strip()] if item['categories'].value.strip() else None,
                                   minimum_inclusive=not item['strict'].value,match=item['match'].value,
                                   case_sensitive=item['case_sensitive'].value)
                    result = result.filter(item['column'].value,**options)
                    filters.append(dict(column=item['column'].value,**options))
            first,second = self.specs()
            # CONFIGURATION LOGIC: Copy all pending analysis choices before constructing any new displayed result.
            inference = InferenceConfig(loss=self.loss.value,unit=self.test_unit.value,correction=self.correction.value,
                                        alpha=self.alpha.value,min_units=self.min_units.value)
            inference = result.inference_config(inference)
            population,min_count,metric = self.candidate_population.value,self.minimum.value,self.metric.value
            temporal = self.temporal_settings()
            trades = self.trade_settings()
            candidate_count = result.coverage['candidate_evaluable'] if population == 'candidate' else len(result.rows)
            if not candidate_count and result.rows.empty:
                raise ValueError(f'No finite candidate records in the selected diagnostic population. Coverage: {result.coverage}')
            title = f'{candidate} vs {reference} | {first.column}'+(f' × {second.column}' if second else '')
            views = self.paired_views(result,first,second,inference,min_count,metric,title)
            views += [self.candidate_view(result,first,second,population,min_count,self.top_n.value),
                      self.temporal_view(result,temporal,population),self.trade_view(result,trades),
                      self.explanation_view(result),_table(result.missingness(),'Input missingness')]
            # UI LOGIC: Publish the complete view and its matching export state only after every calculation succeeds.
            self.views.children = views
            for i,label in enumerate(['Overview','Slices','Dates','Candidate','Time series','Trade points','Feature explanations','Missingness']):
                self.views.set_title(i,label)
            self.result,self.applied_specs = result,(first,second)
            self._refresh_trade_entities(comparison=result)
            self.applied_filters,self.applied_min_count = result.filter_history,min_count
            self.applied_metric,self.applied_inference = metric,inference
            self.applied_candidate_population = population
            self.applied_temporal = temporal
            self.applied_trades = trades
            self.applied_top_n = self.top_n.value
            self._applied_control_state = self._control_state()
            self.export_button.disabled = False
            filters_text = '; '.join(_filter_text(f) for f in result.filter_history) or 'All supplied records'
            self.applied.value = ('<b>Applied:</b> '+escape(title)+f' | {len(result.rows):,} paired records<br><b>Population:</b> '+escape(filters_text)+
                                  f'<br><b>Inference:</b> {inference.loss}, unit={inference.unit}, {inference.correction.upper()}, '
                                  f'alpha={inference.alpha:g}, minimum records={min_count}, minimum units={inference.min_units}; '
                                  f'<b>Candidate population:</b> {population}; '
                                  '<b>Time diagnostics:</b> '+escape(f'{temporal.frequency}, {temporal.signal}' if temporal else 'off'))
            self.status.value = 'Ready. Controls affect the next Apply. Export uses the applied choices. Dependence across dates and repeated exploration still require care.'
        except Exception as exc:
            self.status.value = '<b>Comparison error:</b> '+escape(str(exc))+' Previous applied result is unchanged.'
        finally:
            self._set_busy(False)

    def export(self, _=None):
        # FILE IO LOGIC: Export the applied data/specifications, never newly edited but unapplied controls.
        if self.result is None or self.busy:
            return
        self._set_busy(True,exporting=True)
        try:
            first,second = self.applied_specs
            output = self.result.export(self.export_path.value,slices=[first],interactions=[(first,second)] if second else [],
                                        min_count=self.applied_min_count,metric=self.applied_metric,inference=self.applied_inference,
                                        include_candidate=True,candidate_population=self.applied_candidate_population,candidate_top_n=self.applied_top_n,
                                        temporal=self.applied_temporal,temporal_population=self.applied_candidate_population,trades=self.applied_trades)
            self.status.value = 'Saved complete PNG, CSV and HTML review to '+escape(str(output))
        except Exception as exc:
            self.status.value = '<b>Export error:</b> '+escape(str(exc))
        finally:
            self._set_busy(False)

    def show(self):
        # UI LOGIC: Display one workbench instance; the user explicitly applies a comparison.
        display(self.widget)
        return self


def show_comparison(data=None, **kwargs):
    # UI LOGIC: One public notebook entry for DataFrames, paths or an existing Comparison.
    return ComparisonPanel(data,**kwargs).show()
