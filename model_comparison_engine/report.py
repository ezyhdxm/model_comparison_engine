"""Local reproducible HTML/CSV/PNG review bundles generated directly from comparison results."""
# SETUP LOGIC: Report rendering escapes user-supplied labels and never trains a model.
from datetime import datetime, timezone
from dataclasses import asdict
from html import escape
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import numpy as np
import pandas as pd
from .slices import Slice
from . import plots
from .inference_plots import significance_heatmap
from .diagnostic_plots import candidate_figure, worst_slices_figure
from .diagnostics import candidate_tables
from .ui_style import REPORT_STYLE
from .temporal import TemporalConfig
from .temporal_plots import temporal_figure, event_lag_figure

STYLE = REPORT_STYLE + '''.analysis-report pre{white-space:pre-wrap;overflow-wrap:anywhere}
.analysis-report .figure-scroll{overflow:auto;max-height:1050px}.analysis-report section{scroll-margin-top:20px}'''


def _table(frame):
    # FORMATTING LOGIC: Exact machine-readable values are exported separately; HTML is rounded for reading.
    return '<div class="table-wrap">'+frame.to_html(index=False,escape=True,float_format=lambda x:f'{x:,.5g}')+'</div>'


def _figure(filename, description):
    # PLOTTING LOGIC: Fit the report column while preserving a direct link to the complete full-resolution PNG.
    source, label = escape(filename,quote=True), escape(description,quote=True)
    return (f'<figure><div class="figure-scroll"><a href="{source}" target="_blank" rel="noopener">'
            f'<img src="{source}" alt="{label}"></a></div><figcaption>{label} · '
            f'<a href="{source}" target="_blank" rel="noopener">Open full-size PNG</a></figcaption></figure>')


def _json(value):
    # SERIALIZATION LOGIC: Convert configuration values into portable JSON without arbitrary object repr.
    if isinstance(value, Slice):
        return value.to_dict()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def _portable(value):
    # SERIALIZATION LOGIC: Represent open bin endpoints as strings, keeping the manifest valid standard JSON.
    if isinstance(value,(Slice,np.generic,np.ndarray,Path)):
        return _portable(_json(value))
    if isinstance(value,dict):
        return {str(k):_portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_portable(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        return None if np.isnan(value) else ('inf' if value > 0 else '-inf')
    return value


def _inference_section(table, output, key, title, unit):
    # PLOTTING LOGIC: Export the same significance display used interactively, alongside its exact table.
    significance_heatmap(table, title=title, unit_label=unit).savefig(output/f'{key}.png', dpi=160)
    return _figure(f'{key}.png','Paired loss t statistics and support')+'<details><summary>Exact inference table</summary>'+_table(table)+'</details>'


def _candidate_section(comparison, specs, output, tables, population, min_count, top_n):
    # REPORTING LOGIC: Diagnose the independently evaluable candidate population unless paired is explicitly chosen.
    rows = comparison.candidate_rows(population)
    diagnosis = candidate_tables(rows, specs, min_count=min_count, tolerance=comparison.tolerance,
                                 top_n=top_n, id_column=comparison.config['id_column'])
    diagnosis['summary']['population'] = population
    tables.update({f'candidate_{name}':frame for name,frame in diagnosis.items()})
    candidate_figure(rows, diagnosis, name=comparison.candidate_name, unit=comparison.unit,
                     error_scale=comparison.config['error_scale']).savefig(output/'candidate_diagnostics.png', dpi=160)
    worst_slices_figure(diagnosis['worst_slices'], unit=comparison.unit).savefig(output/'candidate_worst_slices.png', dpi=160)
    section = '<h2 id="candidate">Candidate standalone diagnosis</h2>'
    section += f'<p>Population: <strong>{escape(population)}</strong>; {len(rows):,} evaluable records; '
    section += f'{comparison.coverage["paired"]:,} paired records in the model comparison. '
    section += 'Candidate-only metrics may have a different denominator and must not be used as paired improvement estimates.</p>'
    section += _table(diagnosis['summary'])+_figure('candidate_diagnostics.png','Candidate residual and calibration diagnostics')
    section += _figure('candidate_worst_slices.png','Worst candidate slices, sample support and error contribution')
    section += f'<p>Signed residual = (prediction − actual) × {comparison.config["error_scale"]:g}. Positive residual means overprediction. '
    section += 'Worst groups and cases are descriptive review priorities, not deletion rules. Overlapping slices cannot be summed. '
    section += 'The case export contains record identifiers and selected metadata when supplied; keep the report appropriately private.</p>'
    for key in ['worst_slices','worst_cases','calibration','residual_quantiles','daily']:
        section += '<details><summary>'+escape(key.replace('_',' ').title())+'</summary>'+_table(diagnosis[key])+'</details>'
    return section


def _temporal_section(comparison, output, tables, settings, population, entity):
    # REPORTING LOGIC: Clock gaps, support and change candidates are calculated from the declared saved errors.
    temporal = comparison.temporal_diagnostics(settings, population=population, entity=entity)
    tables.update({f'temporal_{key}':value for key,value in temporal.items()})
    temporal_figure(temporal, name=comparison.candidate_name, unit=comparison.unit).savefig(output/'temporal.png', dpi=160)
    event_lag_figure(temporal['event_autocorrelation'], name=comparison.candidate_name,
                    unit=comparison.unit).savefig(output/'temporal_events.png', dpi=160)
    section = '<h2 id="temporal">Residual timing and candidate change periods</h2>'
    section += '<p class="callout">These exploratory diagnostics use existing predictions. Empty time bins stay missing. '
    section += 'Frequency peaks and change candidates are review prompts, not p-values or model-selection rules. '
    section += 'Changing entity, maturity or trade-size composition can move aggregate residuals; inspect a fixed cohort too.</p>'
    section += _table(temporal['summary'])+_figure('temporal.png','Residual drift, coverage, clock lag correlation and spectrum')
    section += _figure('temporal_events.png','Within-entity event lag correlations and actual time gaps')
    section += '<p>Clock bins use equal elapsed UTC durations; one event lag means a previous distinct timestamp for the same entity. '
    section += 'Simultaneous predictions are averaged only for the event diagnostic. Event-pair correlations still include persistent entity bias '
    section += 'and weight active entities more heavily. Change-point brackets show the last observed bin before and first after a candidate shift.</p>'
    for key in ['change_points','autocorrelation','event_autocorrelation','spectrum','series']:
        label = escape(key.replace('_',' ').title())
        section += f'<details><summary>{label}</summary><p>First 200 rows; <a href="temporal_{key}.csv">download the complete table</a>.</p>'
        section += _table(temporal[key].head(200))+'</details>'
    section += '<p>Methods: <a href="https://arxiv.org/abs/1703.09824">Lomb–Scargle interpretation</a> · '
    section += '<a href="https://arxiv.org/abs/1801.00718">Offline change-point methods</a>.</p>'
    return section


def export_comparison(comparison, folder, slices=None, interactions=None, min_count=30, metric='mae_delta', *,
                      inference=None, include_candidate=True, candidate_population='candidate', candidate_top_n=20,
                      temporal=None, temporal_population=None, temporal_entity=None):
    # CONFIGURATION LOGIC: Resolve explicit inference settings before creating a report directory.
    policy = comparison.inference_config(inference)
    if candidate_population not in {'candidate','paired'}:
        raise ValueError("candidate_population must be 'candidate' or 'paired'.")
    if isinstance(candidate_top_n, bool) or not isinstance(candidate_top_n, (int,np.integer)) or candidate_top_n < 1:
        raise ValueError('candidate_top_n must be a positive integer.')
    temporal = TemporalConfig(**temporal) if isinstance(temporal, dict) else temporal
    if temporal is not None and not isinstance(temporal, TemporalConfig):
        raise TypeError('temporal must be TemporalConfig, a configuration dict or None.')
    temporal_population = candidate_population if temporal_population is None else temporal_population
    if temporal_population not in {'candidate','paired'}:
        raise ValueError("temporal_population must be 'candidate' or 'paired'.")
    # FILE IO LOGIC: Every click creates a separate self-contained review bundle.
    output = Path(folder)/('review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    specs = comparison.default_slices if slices is None else [s if isinstance(s,Slice) else Slice(s) for s in slices]
    interactions = interactions or []
    tables = {'summary':comparison.summary(min_count), 'daily':comparison.daily(min_count),
              'date_sensitivity':comparison.stability(), 'missingness':comparison.missingness(),
              'paired_test':comparison.paired_test(policy, min_count)}
    # PLOTTING LOGIC: Overview and complete date history are exported at full resolution.
    plots.overview(comparison).savefig(output/'overview.png',dpi=160)
    plots.daily_figure(tables['daily'],unit=comparison.unit).savefig(output/'daily.png',dpi=160)
    sections = ['<h2 id="overview">Common-sample overview</h2>'+_figure('overview.png','Common-sample metrics and coverage')
                +'<details><summary>Exact overall metrics</summary>'+_table(tables['summary'])+'</details>']
    sections.append('<h2 id="inference">Overall paired loss test</h2>'+_table(tables['paired_test']))
    # REPORTING LOGIC: Build every requested slice from the same result object used in the notebook.
    for i,spec in enumerate(specs):
        key = f'slice_{i+1:02d}'
        table = comparison.slice(spec,min_count=min_count)
        tables[key] = table
        title = spec.name or spec.column
        plots.slice_figure(table,metric,unit=comparison.unit,title=title).savefig(output/f'{key}.png',dpi=160)
        sections.append(f'<h2 id="{key}">{escape(title)}</h2>'+_figure(f'{key}.png','Slice loss and support')+'<details><summary>Complete table</summary>'+_table(table)+'</details>')
        tables[key+'_test'] = comparison.slice_test(spec, min_count=min_count, inference=policy)
        sections.append(_inference_section(tables[key+'_test'], output, key+'_test', title, comparison.unit))
    # REPORTING LOGIC: Cross tables preserve every observed cell and its support.
    for i,pair in enumerate(interactions):
        sx,sy = [s if isinstance(s,Slice) else Slice(s) for s in pair]
        key = f'interaction_{i+1:02d}'
        table = comparison.cross_slice(sx,sy,min_count=min_count)
        tables[key] = table
        if not table.empty:
            plots.heatmap(table,metric,unit=comparison.unit,title=f'{sx.column} × {sy.column}').savefig(output/f'{key}.png',dpi=160)
            sections.append(f'<h2>{escape(sx.column)} × {escape(sy.column)}</h2>'+_figure(f'{key}.png','Loss and count heatmaps')+_table(table))
        tables[key+'_test'] = comparison.cross_slice_test(sx, sy, min_count=min_count, inference=policy)
        sections.append(_inference_section(tables[key+'_test'], output, key+'_test', f'{sx.column} × {sy.column}', comparison.unit))
    # REPORTING LOGIC: Include axes requested only through interactions in the standalone weakness review too.
    if include_candidate:
        candidate_specs = list(specs)
        seen_specs = {json.dumps(_portable(s), sort_keys=True) for s in candidate_specs}
        for pair in interactions:
            for value in pair:
                spec = value if isinstance(value, Slice) else Slice(value)
                identity = json.dumps(_portable(spec), sort_keys=True)
                if identity not in seen_specs:
                    candidate_specs.append(spec)
                    seen_specs.add(identity)
        sections.append(_candidate_section(comparison, candidate_specs, output, tables, candidate_population, min_count, candidate_top_n))
    if temporal is not None:
        sections.append(_temporal_section(comparison, output, tables, temporal, temporal_population, temporal_entity))
    # FILE IO LOGIC: Lossless CSV tables and the explicit review configuration accompany the figures.
    for name,table in tables.items():
        table.to_csv(output/f'{name}.csv',index=False)
    hashed = pd.util.hash_pandas_object(comparison.data,index=True,categorize=True).to_numpy().tobytes()
    schema = repr([(str(c),str(t)) for c,t in comparison.data.dtypes.items()]).encode()
    digest = hashlib.sha256(schema+hashed).hexdigest()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),config=comparison.config,
                    coverage=comparison.coverage,data_sha256=digest,slices=specs,
                    interactions=interactions,min_count=min_count,metric=metric,training_performed=False)
    manifest['filters'] = comparison.filter_history
    manifest['inference'] = asdict(policy)
    manifest['inference_family'] = 'Eligible tests within each individual slice/intersection table; overall test is separate.'
    manifest['candidate_diagnostics'] = dict(included=bool(include_candidate), population=candidate_population,
                                            case_limit=candidate_top_n, worst_slice_limit=candidate_top_n, calibration_bins=10)
    manifest['temporal_diagnostics'] = dict(included=temporal is not None, settings=asdict(temporal) if temporal else None,
                                           population=temporal_population, entity=temporal_entity)
    (output/'review.json').write_text(json.dumps(_portable(manifest),allow_nan=False,indent=2),encoding='utf-8')
    # REPORTING LOGIC: State denominators, signs and descriptive limitations next to the generated evidence.
    title = f'{comparison.candidate_name} vs {comparison.reference_name}'
    intro = '<header class="hero"><span class="eyebrow">Model comparison engine · evidence review</span>'
    intro += f'<h1>{escape(title)}</h1><p>Errors in {escape(comparison.unit)} · {comparison.coverage["paired"]:,} common records '
    intro += f'· {comparison.coverage["total"]:,} supplied records</p></header>'
    intro += '<nav aria-label="Report sections"><a href="#overview">Overview</a><a href="#inference">Paired inference</a>'
    if specs:
        intro += '<a href="#slice_01">Slices</a>'
    if include_candidate:
        intro += '<a href="#candidate">Candidate diagnosis</a>'
    if temporal is not None:
        intro += '<a href="#temporal">Residual timing</a>'
    intro += '<a href="#dates">Dates</a></nav><details><summary>How to read this review</summary>'
    intro += '<p class="note">Record-weighted metrics use common finite targets and predictions. Negative MAE/P95 delta means improvement; positive MAE improvement % means improvement. '
    intro += 'Sparse cells are flagged, never removed. Slices are descriptive, overlap, and must not be added together. '
    intro += 'Date sensitivity is not a confidence interval. Repeated test inspection is not fresh validation.</p>'
    intro += f'<p class="note">Paired inference: {escape(policy.loss)} loss; unit={escape(policy.unit)}; '
    intro += f'correction={escape(policy.correction)} within each test table; alpha={policy.alpha:g}; minimum units={policy.min_units}. '
    intro += 'Negative t means lower candidate loss. Date/entity tests average differences within each unit first and then weight units equally; '
    intro += 'this differs from record-weighted MAE. Tests assume independent units; date grouping does not resolve serial dependence, '
    intro += 'and false-discovery correction does not make reused data a fresh holdout. '
    intro += 'Stars mark adjusted p-value thresholds (.05/.01/.001), while † marks low support. '
    intro += 'Low support or degenerate variance is untested, not evidence of equivalence. Confidence intervals are pointwise, not simultaneous.</p>'
    intro += '</details><details><summary>Full coverage audit</summary><pre>'+escape(json.dumps(comparison.coverage,indent=2))+'</pre></details>'
    tail = '<section><h2 id="dates">Dates and sensitivity</h2>'+_figure('daily.png','Daily losses')+_table(tables['date_sensitivity'])
    tail += '<p><a href="review.json">Configuration and data fingerprint</a> · <a href="summary.csv">Exact summary</a></p>'
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title><style>'+STYLE+'</style><body class="analysis-report"><main>'
    (output/'report.html').write_text(document+intro+''.join('<section>'+s+'</section>' for s in sections)+tail+'</section></main></body></html>',encoding='utf-8')
    return output
