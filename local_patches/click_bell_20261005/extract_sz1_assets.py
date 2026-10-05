"""CPU-only extraction of the existing SZ1 click_bell clean50 start states.

Only write beneath OUTPUT. Input HDF5/instructions/seeds remain unchanged.
Run through the existing fixed-host-key SSH using the established RLinf Python.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys


SOURCE = Path('/data/chenyiteng/datasets/robotwin2/raw/9dc9299c163db059931898a9f0852098a61155a1/click_bell/clean50-20261002/aloha-agilex_clean_50')
OUTPUT = Path('/data/chenyiteng/projects/wmrl-click-bell-assets-20261005')
TASK = 'click_bell'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    if not path.parent.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError('Output outside the authorized extraction directory')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--builder', type=Path, default=OUTPUT / 'code/build_reset_data.py')
    p.add_argument('--native-seed-source', type=Path)
    a = p.parse_args()
    if os.environ.get('USER') != 'chenyiteng' or Path.home().name != 'chenyiteng':
        raise RuntimeError('Run as the established chenyiteng account')
    if not a.builder.resolve().is_relative_to(OUTPUT.resolve()) or not a.builder.is_file():
        raise ValueError('Require the staged CPU reset builder in the authorized directory')
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reset = OUTPUT / 'click_bell_clean50_reset.npz'
    complete = OUTPUT / 'complete.json'
    if complete.exists():
        previous = json.loads(complete.read_text())
        for row in previous['files']:
            target = OUTPUT / row['relative_path']
            if not target.is_file() or target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                raise RuntimeError('Previous extraction receipt no longer matches its output')
        print(json.dumps(dict(status='already_complete', **previous)))
        return
    if reset.exists() or reset.with_suffix('.json').exists():
        raise RuntimeError('Incomplete prior extraction exists; inspect it rather than overwrite')
    native_value = None
    if a.native_seed_source is not None:
        seed_input = a.native_seed_source.resolve()
        if not seed_input.is_relative_to(Path('/data/chenyiteng').resolve()) and not seed_input.is_relative_to(Path('/home/chenyiteng').resolve()):
            raise ValueError('Seed source must be an explicit file owned by our project')
        if seed_input.stat().st_size > 2 * 1024 * 1024:
            raise ValueError('Unexpectedly large native seed file')
        seed_data = json.loads(seed_input.read_text())
        native_value = seed_data.get(TASK)
        values = native_value.get('success_seeds') if isinstance(native_value, dict) else None
        if not isinstance(values, list) or len(values) < 32 or any(type(v) is not int for v in values) or len(set(values)) != len(values):
            raise ValueError('Explicit seed source lacks an existing click_bell list of at least 32 unique integers')
    argv = sys.argv
    try:
        sys.argv = [str(a.builder), '--source', str(SOURCE), '--output', str(reset), '--task-name', TASK]
        runpy.run_path(str(a.builder), run_name='__main__')
    finally:
        sys.argv = argv
    import numpy as np
    with np.load(reset, allow_pickle=False) as data:
        episodes = data['reset_ids'].tolist()
        instructions = data['instructions'].tolist()
        shapes = {name: list(data[name].shape) for name in data.files}
    if len(episodes) != 50:
        raise RuntimeError('Builder did not produce complete clean50')
    # These are dataset seed records only. Never reinterpret episode0 as seed0,
    # or relabel another task's native evaluation seeds as click_bell.
    seed_records = []
    seed_mapping = None
    for name in ('seed.txt', 'seeds.txt', 'seed.json', 'seeds.json'):
        path = SOURCE / name
        if not path.is_file():
            continue
        if path.stat().st_size > 1024 * 1024:
            raise ValueError('Unexpectedly large dataset seed metadata')
        content = path.read_text(encoding='utf-8')
        record = {'source': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path), 'text': content}
        if path.suffix == '.txt':
            tokens = content.split()
            if len(tokens) == 50 and all(re.fullmatch(r'[0-9]+', token) for token in tokens):
                record['integer_list'] = [int(token) for token in tokens]
                record['mapping_status'] = 'ordered_raw_seed_file; no independent collection-log verification'
                if seed_mapping is not None and seed_mapping != record['integer_list']:
                    raise RuntimeError('Dataset seed metadata files disagree')
                seed_mapping = record['integer_list']
        seed_records.append(record)
    rows = []
    for i, (episode, instruction) in enumerate(zip(episodes, instructions)):
        instruction_path = SOURCE / 'instructions' / (episode + '.json')
        rows.append({'reset_id': i, 'episode': episode, 'instruction': instruction,
                     'instruction_source': str(instruction_path), 'instruction_source_sha256': sha(instruction_path),
                     'dataset_seed_from_ordered_raw_record': None if seed_mapping is None else seed_mapping[i]})
    metadata = {'task': TASK, 'source': str(SOURCE), 'count': 50, 'selection': 'all existing episodes, no filtering',
                'seed_mapping_status': 'not_available' if seed_mapping is None else 'ordered_raw_seed_file; not independently verified',
                'dataset_seed_records': seed_records, 'rows': rows, 'array_shapes': shapes,
                'native_evaluation_seeds': 'separate task seed source required; never infer from episode index'}
    save(OUTPUT / 'instructions-and-seeds.json', metadata)
    output_files = [reset, reset.with_suffix('.json'), OUTPUT / 'instructions-and-seeds.json']
    if native_value is not None:
        destination = OUTPUT / 'click_bell_native_seeds.json'
        save(destination, {TASK: native_value})
        seed_receipt = OUTPUT / 'click_bell_native_seeds.receipt.json'
        save(seed_receipt, {'source': str(a.native_seed_source), 'source_sha256': sha(a.native_seed_source),
                            'task': TASK, 'count': len(native_value['success_seeds']),
                            'selection': 'existing task entry copied unchanged; no generated or filtered seeds',
                            'output_sha256': sha(destination)})
        output_files.extend([destination, seed_receipt])
    receipt = {'schema': 1, 'time_utc': datetime.now(timezone.utc).isoformat(), 'task': TASK, 'source': str(SOURCE),
               'output': str(OUTPUT), 'count': 50, 'cpu_only': True, 'source_data_modified': False,
               'builder_sha256': sha(a.builder), 'script_sha256': sha(__file__),
               'files': [{'relative_path': str(path.relative_to(OUTPUT)), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in output_files]}
    save(complete, receipt)
    print(json.dumps(dict(status='complete', **receipt)))


if __name__ == '__main__':
    main()
