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

STYLE = '''body{max-width:1120px;margin:35px auto;padding:0 22px;font:16px/1.65 system-ui;color:#183541}
h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border:1px solid #d5e0e4;text-align:left}
th{background:#edf4f5}.table{overflow:auto}img{max-width:100%}.note{background:#edf7f5;padding:16px;border-left:4px solid #18807d}
code{overflow-wrap:anywhere}a{color:#097986}section{margin:32px 0}details{margin:15px 0}'''


def _table(frame):
    # FORMATTING LOGIC: Exact machine-readable values are exported separately; HTML is rounded for reading.
    return '<div class="table">'+frame.to_html(index=False,escape=True,float_format=lambda x:f'{x:,.5g}')+'</div>'


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
    return f'<img src="{key}.png" alt="Paired loss t statistics and support"><details><summary>Exact inference table</summary>'+_table(table)+'</details>'


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
    section = '<h2>Candidate standalone diagnosis</h2>'
    section += f'<p>Population: <strong>{escape(population)}</strong>; {len(rows):,} evaluable records; '
    section += f'{comparison.coverage["paired"]:,} paired records in the model comparison. '
    section += 'Candidate-only metrics may have a different denominator and must not be used as paired improvement estimates.</p>'
    section += _table(diagnosis['summary'])+'<img src="candidate_diagnostics.png" alt="Candidate residual and calibration diagnostics">'
    section += '<img src="candidate_worst_slices.png" alt="Worst candidate slices, sample support and error contribution">'
    section += f'<p>Signed residual = (prediction − actual) × {comparison.config["error_scale"]:g}. Positive residual means overprediction. '
    section += 'Worst groups and cases are descriptive review priorities, not deletion rules. Overlapping slices cannot be summed. '
    section += 'The case export contains record identifiers and selected metadata when supplied; keep the report appropriately private.</p>'
    for key in ['worst_slices','worst_cases','calibration','residual_quantiles','daily']:
        section += '<details><summary>'+escape(key.replace('_',' ').title())+'</summary>'+_table(diagnosis[key])+'</details>'
    return section


def export_comparison(comparison, folder, slices=None, interactions=None, min_count=30, metric='mae_delta', *,
                      inference=None, include_candidate=True, candidate_population='candidate', candidate_top_n=20):
    # CONFIGURATION LOGIC: Resolve explicit inference settings before creating a report directory.
    policy = comparison.inference_config(inference)
    if candidate_population not in {'candidate','paired'}:
        raise ValueError("candidate_population must be 'candidate' or 'paired'.")
    if isinstance(candidate_top_n, bool) or not isinstance(candidate_top_n, (int,np.integer)) or candidate_top_n < 1:
        raise ValueError('candidate_top_n must be a positive integer.')
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
    sections = ['<h2>Common-sample overview</h2>'+_table(tables['summary'])+'<img src="overview.png" alt="Common-sample metrics and coverage">']
    sections.append('<h2>Overall paired loss test</h2>'+_table(tables['paired_test']))
    # REPORTING LOGIC: Build every requested slice from the same result object used in the notebook.
    for i,spec in enumerate(specs):
        key = f'slice_{i+1:02d}'
        table = comparison.slice(spec,min_count=min_count)
        tables[key] = table
        title = spec.name or spec.column
        plots.slice_figure(table,metric,unit=comparison.unit,title=title).savefig(output/f'{key}.png',dpi=160)
        sections.append(f'<h2>{escape(title)}</h2><img src="{key}.png" alt="Slice loss and support"><details><summary>Complete table</summary>'+_table(table)+'</details>')
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
            sections.append(f'<h2>{escape(sx.column)} × {escape(sy.column)}</h2><img src="{key}.png" alt="Loss and count heatmaps">'+_table(table))
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
    (output/'review.json').write_text(json.dumps(_portable(manifest),allow_nan=False,indent=2),encoding='utf-8')
    # REPORTING LOGIC: State denominators, signs and descriptive limitations next to the generated evidence.
    title = f'{comparison.candidate_name} vs {comparison.reference_name}'
    intro = f'<h1>{escape(title)}</h1><p>Errors in {escape(comparison.unit)}. Record-weighted metrics on common finite targets and predictions.</p>'
    intro += '<p class="note">Negative MAE/P95 delta means improvement; positive MAE improvement % means improvement. '
    intro += 'Sparse cells are flagged, never removed. Slices are descriptive, overlap, and must not be added together. '
    intro += 'Date sensitivity is not a confidence interval. Repeated test inspection is not fresh validation.</p>'
    intro += f'<p class="note">Paired inference: {escape(policy.loss)} loss; unit={escape(policy.unit)}; '
    intro += f'correction={escape(policy.correction)} within each test table; alpha={policy.alpha:g}; minimum units={policy.min_units}. '
    intro += 'Negative t means lower candidate loss. Date/entity tests average differences within each unit first and then weight units equally; '
    intro += 'this differs from record-weighted MAE. Tests assume independent units; date grouping does not resolve serial dependence, '
    intro += 'and false-discovery correction does not make reused data a fresh holdout. '
    intro += 'Stars mark adjusted p-value thresholds (.05/.01/.001), while † marks low support. '
    intro += 'Low support or degenerate variance is untested, not evidence of equivalence. Confidence intervals are pointwise, not simultaneous.</p>'
    intro += '<h2>Coverage</h2><pre>'+escape(json.dumps(comparison.coverage,indent=2))+'</pre>'
    tail = '<h2>Dates and sensitivity</h2><img src="daily.png" alt="Daily losses">'+_table(tables['date_sensitivity'])
    tail += '<p><a href="review.json">Configuration and data fingerprint</a> · <a href="summary.csv">Exact summary</a></p>'
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>'+escape(title)+'</title><style>'+STYLE+'</style><body>'
    (output/'report.html').write_text(document+intro+''.join('<section>'+s+'</section>' for s in sections)+tail+'</body></html>',encoding='utf-8')
    return output
