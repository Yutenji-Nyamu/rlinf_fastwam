"""CPU-only, one-shot budget extension; preserve the immutable 20k baseline."""
import ast
import copy
import json
import os
import shutil
import socket
import time
from common import *

PAYLOAD = {'base', 'core', 'replay', 'cadence', 'rng', 'progress'}

def extend_metadata(saved):
    assert saved['cadence']['contract']['max_physical_actions'] == 20000
    c = saved['cadence']['counters']
    assert c['physical_actions'] == 20000 and c['episodes_completed'] == 128
    assert c['completed_calls'] == 454 and c['pending_calls'] == 0 and c['carry_actions'] == 13
    assert c['budget_truncated_episodes'] == 1
    saved['cadence']['contract'].update(max_physical_actions=60000, prior_budget_truncations=1)
    saved['progress']['evaluation_final'] = False

def check_inputs(old, new):
    expected = copy.deepcopy(old)
    assert expected['formal']['max_physical_actions'] == 20000
    expected['formal']['max_physical_actions'] = 60000
    old_pins = expected.pop('port_source_manifest')
    actual = copy.deepcopy(new); new_pins = actual.pop('port_source_manifest')
    assert actual == expected and set(old_pins) == set(new_pins)
    assert {p for p in old_pins if old_pins[p] != new_pins[p]} == CHANGED_SOURCE

def main():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    run = TRAIN / 'run'; backup = CONTROL / 'baseline-20000'
    assert read(run / 'complete.json')['ok'] is True
    assert read(ROOT / 'eval10-continuation-20261003/final.json')['rlt_first_rounds_verified'] is True
    assert not backup.exists() and not (CONTROL / 'migration.json').exists()
    old = read(TRAIN / 'inputs.json'); resolved = read(run / 'resolved.json')
    old_contract = resolved['contract']
    assert resolved['inputs'] == old and sha(TRAIN / 'inputs.json') == old_contract['inputs_sha256']
    assert resolved['contract_sha256'] == jhash(old_contract)
    for name, pin in old['port_source_manifest'].items():
        assert sha(TRAIN / 'source' / name) == pin
        if name not in CHANGED_SOURCE: assert sha(SOURCE / name) == pin
    new = copy.deepcopy(old); new['formal']['max_physical_actions'] = 60000
    new['port_source_manifest'].update({p: sha(SOURCE / p) for p in CHANGED_SOURCE})
    check_inputs(old, new)
    new_bytes = (json.dumps(new, indent=2, allow_nan=False) + '\n').encode()
    import hashlib
    contract = dict(old_contract, max_physical_actions=60000, inputs_sha256=hashlib.sha256(new_bytes).hexdigest())
    tree = ast.parse((SOURCE / 'examples/embodiment/train_expo_ft.py').read_text())
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'digest']
    ns = {'torch': torch, 'hashlib': hashlib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'digest', 'exec'), ns)
    digest = ns['digest']
    backup.mkdir(mode=0o700)
    # Link immutable large payloads, copy mutable metadata. Never edit these backups.
    for folder in ('run', 'replay'):
        for p in (TRAIN / folder).rglob('*'):
            assert not p.is_symlink()
            target = backup / p.relative_to(TRAIN)
            if p.is_dir(): target.mkdir(parents=True, exist_ok=True)
            elif p.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                if p.suffix == '.pt': os.link(p, target)
                else: shutil.copy2(p, target)
    shutil.copy2(TRAIN / 'inputs.json', backup / 'inputs.json')
    replay_before = sha(TRAIN / 'replay/index.json')
    results = {}
    for name in ('checkpoint-latest.pt', 'checkpoint-last1.pt'):
        saved = torch.load(backup / 'run' / name, map_location='cpu', weights_only=False)
        assert saved['contract'] == old_contract and saved['contract_sha256'] == jhash(old_contract)
        hashes = {k: digest(saved[k]) for k in PAYLOAD}
        assert saved['hashes'] == hashes
        assert saved['replay']['online_entries'] == read(TRAIN / 'replay/index.json')['online_entries']
        before_counters = copy.deepcopy(saved['cadence']['counters'])
        before_progress = copy.deepcopy(saved['progress'])
        extend_metadata(saved)
        assert saved['cadence']['counters'] == before_counters
        expected_progress = dict(before_progress, evaluation_final=False)
        assert saved['progress'] == expected_progress
        after = {k: digest(saved[k]) for k in PAYLOAD}
        assert all(after[k] == hashes[k] for k in ('base', 'core', 'replay', 'rng'))
        saved.update(contract=contract, contract_sha256=jhash(contract), hashes=after)
        target = CONTROL / name
        with target.open('xb') as f:
            torch.save(saved, f); f.flush(); os.fsync(f.fileno())
        del saved
        verified = torch.load(target, map_location='cpu', weights_only=False)
        assert {k: digest(verified[k]) for k in PAYLOAD} == after
        assert verified['contract'] == contract
        results[name] = {'before': hashes, 'after': after, 'cadence': verified['cadence'], 'bytes': target.stat().st_size}
        del verified
    assert sha(TRAIN / 'replay/index.json') == replay_before
    temp = TRAIN / 'inputs.json.60k-partial'; temp.write_bytes(new_bytes); temp.replace(TRAIN / 'inputs.json')
    for name in results: (CONTROL / name).replace(run / name)
    new_resolved = dict(resolved, inputs=new, contract=contract, contract_sha256=jhash(contract))
    new_resolved['cli'] = dict(resolved['cli'], max_physical_actions=60000)
    atomic(run / 'resolved.json', new_resolved)
    cp = read(backup / 'run/checkpoint.json')
    cp.update(bytes=results['checkpoint-latest.pt']['bytes'], hashes=results['checkpoint-latest.pt']['after'],
              cadence=results['checkpoint-latest.pt']['cadence'], reason='authorized-budget-extension-20000-to-60000')
    atomic(run / 'checkpoint.json', cp)
    # The original terminal receipts remain in the baseline snapshot; old evaluations stay in place.
    for name in ('complete.json', 'stopped.json', 'failure.json'):
        p = run / name
        if p.exists():
            assert (backup / 'run' / name).read_bytes() == p.read_bytes()
            p.unlink()
    result = dict(ok=True, time=time.time(), old_budget=20000, new_budget=60000, additional_actions=40000,
                  old_inputs_sha256=old_contract['inputs_sha256'], new_inputs_sha256=sha(TRAIN / 'inputs.json'),
                  checkpoint_results=results, unchanged_payloads=['base', 'core', 'replay', 'rng'],
                  counters_unchanged=True, evaluation_history_retained=True, warmup_restarted=False,
                  replay_index_sha256=replay_before, cpu_only=not torch.cuda.is_initialized(), baseline=str(backup))
    atomic(CONTROL / 'migration.json', result)
    print(json.dumps(result))

if __name__ == '__main__': main()
