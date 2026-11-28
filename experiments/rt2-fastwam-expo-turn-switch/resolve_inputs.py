"""Resolve machine-specific paths for the two-card FastWAM EXPO recipe."""
import argparse
import json
import os
from pathlib import Path
from string import Template

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
values = dict(os.environ, TURN_SWITCH_SEEDS=str(root / 'seeds'))
def fill(value):
    if isinstance(value, list):
        return [fill(item) for item in value]
    if isinstance(value, dict):
        return {key: fill(item) for key, item in value.items()}
    if isinstance(value, str) and value.startswith('$'):
        return Template(value).substitute(values)
    return value
cfg = json.loads((root / 'inputs.template.json').read_text(encoding='utf-8'))
args.output.write_text(json.dumps(fill(cfg), indent=2) + '\n', encoding='utf-8')
