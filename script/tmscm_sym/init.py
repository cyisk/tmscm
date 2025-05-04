import argparse
import yaml
from typing import Dict, Any

parser = argparse.ArgumentParser(
    description="Pipeline for TMSCM-SYM experiments")
parser.add_argument(
    '--name',
    '-n',
    type=str,
    help='Name of this experiment',
    required=True,
)
parser.add_argument(
    '--dataset',
    '-d',
    type=str,
    help='Dataset of TMSCM-SYM',
    choices=['barbell', 'stair', 'fork', 'backdoor'],
    required=True,
)
parser.add_argument(
    '--exogenous_distribution',
    '-e',
    type=str,
    help='Type of exogenous distribution',
    choices=['n', 'nf', 'gmm'],
    default='nf',
    required=False,
)
parser.add_argument(
    '--non_markovian',
    '-nm',
    action='store_true',
    help='Ablation experiments; non-markovian exogenous distribution',
    required=False,
)
parser.add_argument(
    '--solution_mapping',
    '-s',
    type=str,
    help='Type of solution mapping',
    choices=['dnme', 'tnme', 'cmsm', 'tvsm'],
    default='dnme',
    required=False,
)
parser.add_argument(
    '--non_causal_ordered',
    '-nc',
    action='store_true',
    help='Ablation experiments; no use of causal order',
    required=False,
)
parser.add_argument(
    '--non_triangular',
    '-nt',
    action='store_true',
    help='Ablation experiments; non-triangular solution mapping',
    required=False,
)
args = parser.parse_args()


def get_experiment_config_string(arguments: Dict[str, Any], flag: str) -> str:
    with open('script/tmscm_sym/config.yaml', mode='r', encoding='utf-8') as f:
        template_string = f.read()

    # Replace template placeholders ("${...}") with arguments
    config_str = template_string
    for key, value in arguments.items():
        config_str = config_str.replace(r'${' + key + r'}', str(value))
    config_str = config_str.replace(r'${experiment_flag}', flag)

    return config_str


def initialize_experiment_metadata() -> Dict[str, str | bool]:
    # Experiment arguments
    arguments = {
        'experiment_name': args.name,
        'dataset': args.dataset,
        'exogenous_distribution': args.exogenous_distribution,
        'markovian': not args.non_markovian,
        'solution_mapping': args.solution_mapping,
        'causal_ordered': not args.non_causal_ordered,
        'triangular': not args.non_triangular,
    }

    # Experiment flag
    flag = '-'.join([
        f"d={arguments['dataset']}",
        f"e={arguments['exogenous_distribution']}",
        f"m={'1' if arguments['markovian'] else '0'}",
        f"s={arguments['solution_mapping']}",
        f"c={'1' if arguments['causal_ordered'] else '0'}",
        f"t={'1' if arguments['triangular'] else '0'}",
    ])

    # Experiment config
    config_string = get_experiment_config_string(arguments, flag)
    config = yaml.safe_load(config_string)

    return {
        'flag': flag,
        'arguments': arguments,
        'config': config,
        'work_dirpath': config['pipeline']['work_dirpath'],
    }
