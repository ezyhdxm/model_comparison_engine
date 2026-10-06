"""Generate a comparison report from saved predictions or a synthetic demonstration."""
# SETUP LOGIC: The command line reuses the public comparison API without training or loading serialized models.
import argparse
import json
from pathlib import Path
from .engine import compare_predictions
from .slices import Slice
from .inference import InferenceConfig


def _spec(value):
    # CONFIGURATION LOGIC: Accept a column name or an explicit slice definition; unknown options are errors.
    return Slice(value) if isinstance(value, str) else Slice(**value)


def run_config(path, output=None):
    # FILE IO LOGIC: Relative paths resolve beside the configuration file, including on other operating systems.
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    data_path = path.parent/Path(config['data_path']).expanduser()
    output = Path(output).expanduser() if output else path.parent/Path(config.get('output', 'reports')).expanduser()
    # CONFIGURATION LOGIC: The caller explicitly chooses prediction/target columns, units and grouping fields.
    options = config['comparison']
    slices = [_spec(value) for value in config['slices']] if 'slices' in config else None
    interactions = [tuple(_spec(value) for value in pair) for pair in config.get('interactions', [])]
    if any(len(pair) != 2 for pair in interactions):
        raise ValueError('Each interaction must specify exactly two slice columns.')
    # REPORTING LOGIC: All tables and figures come from the current comparison, never hand-entered results.
    result = compare_predictions(data_path, **options)
    return result.export(output, slices=slices, interactions=interactions,
                         min_count=config.get('min_count', 30), metric=config.get('metric', 'mae_delta'),
                         inference=InferenceConfig(**config.get('inference', {})),
                         include_candidate=config.get('include_candidate', True),
                         candidate_population=config.get('candidate_population', 'candidate'),
                         candidate_top_n=config.get('candidate_top_n', 20))


def main():
    # UI LOGIC: File-based comparison and synthetic demonstration have separate explicit commands.
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    compare = commands.add_parser('compare', help='Compare two columns of saved scalar predictions')
    compare.add_argument('--config', required=True)
    compare.add_argument('--output')
    demo = commands.add_parser('demo', help='Run synthetic data; results do not establish real-world gains')
    demo.add_argument('--output', default='reports/synthetic_demo')
    args = parser.parse_args()
    # ORCHESTRATION LOGIC: The demo only generates synthetic inputs and invokes the same documented API.
    if args.command == 'demo':
        from .demo import make_demo
        data = make_demo()
        result = compare_predictions(data, 'actual', 'reference_prediction', 'candidate_prediction',
            reference_name='Synthetic reference', candidate_name='Synthetic candidate',
            id_column='row_id', time_column='time', entity_column='entity_id')
        output = result.export(args.output, slices=['segment', 'category'], interactions=[('segment', 'category')])
    else:
        output = run_config(args.config, args.output)
    # UI LOGIC: Print only the report path; full tables are exported instead of filling the console.
    print(f'Open {output / "report.html"}')
