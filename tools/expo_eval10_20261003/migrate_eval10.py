"""One-shot SZ2 eval25->10 metadata migration; all six training payloads stay exact.

Only run under the reviewed handoff controller after the old driver is stopped.
Original checkpoints are hard-linked for rollback, not a new retention policy.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002')
TRANSITION = ROOT.parent / 'eval10-continuation-20261003'
PAYLOAD = {'base', 'core', 'replay', 'cadence', 'rng', 'progress'}
DRIVER = 'examples/embodiment/train_expo_formal.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.eval10-partial')
    with tmp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def check_input_diff(old, new):
    expected = copy.deepcopy(old)
    require(expected['evaluation']['every_episodes'] == 25, 'Original interval is not 25')
    expected['evaluation']['every_episodes'] = 10
    old_pins = expected.pop('port_source_manifest')
    actual = copy.deepcopy(new); pins = actual.pop('port_source_manifest')
    require(actual == expected, 'Training recipe changed outside evaluation interval')
    require(set(pins) == set(old_pins), 'Unexpected source manifest additions/removals')
    require({k for k in old_pins if pins[k] != old_pins[k]} == {DRIVER}, 'Protected training source changed')


def load_digest(torch):
    path = ROOT / 'source/examples/embodiment/train_expo_ft.py'
    tree = ast.parse(path.read_text())
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'digest']
    namespace = {'torch': torch, 'hashlib': hashlib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['digest']


def migrate():
    require(os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02', 'SZ2 identity differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only migration required')
    import torch
    torch.set_num_threads(4)
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    backup = TRANSITION / 'backup'
    backup.mkdir(mode=0o700, exist_ok=False)
    source = ROOT / 'source'; run = ROOT / 'run'
    old = read(ROOT / 'inputs.json'); resolved = read(run / 'resolved.json')
    replay_index_sha = sha(ROOT / 'replay/index.json')
    old_contract = resolved['contract']
    require(old_contract['inputs_sha256'] == sha(ROOT / 'inputs.json')
            and resolved['inputs'] == old and resolved['contract_sha256'] == jhash(old_contract),
            'Resolved inputs/contract mismatch')
    for name, fingerprint in old['port_source_manifest'].items():
        require(sha(source / name) == fingerprint, 'Original source differs: ' + name)
    original_driver = (source / DRIVER).read_bytes()
    patched_driver = (TRANSITION / 'train_expo_formal.py').read_bytes()
    expected = original_driver.replace(b"evaluation.get('every_episodes') != 25", b"evaluation.get('every_episodes') != 10")
    expected = expected.replace(b'Fixed initial/25-episode/final evaluation contract differs',
                                b'Fixed initial/10-episode/final evaluation contract differs')
    require(patched_driver == expected and expected != original_driver, 'Driver patch exceeds two reviewed literals')
    new = copy.deepcopy(old)
    new['evaluation']['every_episodes'] = 10
    new['port_source_manifest'][DRIVER] = hashlib.sha256(patched_driver).hexdigest()
    check_input_diff(old, new)
    new_bytes = (json.dumps(new, indent=2, allow_nan=False) + '\n').encode()
    new_contract = dict(old_contract, inputs_sha256=hashlib.sha256(new_bytes).hexdigest())
    digest = load_digest(torch)
    metadata = ['inputs.json', 'run/resolved.json', 'run/checkpoint.json', 'current.json']
    for name in metadata:
        target = backup / name; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write((ROOT / name).read_bytes())
    with (backup / 'train_expo_formal.py').open('xb') as f:
        f.write(original_driver)
    results = {}
    for filename in ('checkpoint-latest.pt', 'checkpoint-last1.pt'):
        cp = run / filename
        require(cp.is_file() and not cp.is_symlink(), 'Missing/symlinked checkpoint ' + filename)
        os.link(cp, backup / filename)
    atomic_json(TRANSITION / 'backup-ready.json', {'ok': True, 'time': time.time(), 'metadata': metadata})
    for filename in ('checkpoint-latest.pt', 'checkpoint-last1.pt'):
        cp = backup / filename
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        require(saved['version'] == 4 and saved['contract'] == old_contract
                and saved['contract_sha256'] == jhash(old_contract), 'Checkpoint contract differs')
        require(set(saved['hashes']) == PAYLOAD, 'Payload schema differs')
        hashes = {key: digest(saved[key]) for key in PAYLOAD}
        require(hashes == saved['hashes'], 'Original full payload hashes differ')
        if filename == 'checkpoint-latest.pt':
            receipt = read(run / 'checkpoint.json')
            require(receipt['hashes'] == hashes and receipt['cadence'] == saved['cadence'], 'Latest receipt differs')
            require(read(ROOT / 'replay/index.json')['online_entries'] == saved['replay']['online_entries'],
                    'Uncommitted replay episode exists')
            require(len(saved['replay']['online_entries']) == saved['cadence']['counters']['episodes_completed'],
                    'Episode/replay counters differ')
            stopped = read(run / 'stopped.json')
            require(stopped['cadence'] == saved['cadence'], 'Stopped driver/latest boundary differs')
            require(saved['progress']['stopped_episodes'] == 0, 'Stop-truncated collection was recorded')
        saved['contract'] = new_contract; saved['contract_sha256'] = jhash(new_contract)
        destination = TRANSITION / filename
        with destination.open('xb') as f:
            torch.save(saved, f); f.flush(); os.fsync(f.fileno())
        counters = copy.deepcopy(saved['cadence']['counters'])
        del saved
        verified = torch.load(destination, map_location='cpu', weights_only=False)
        require({key: digest(verified[key]) for key in PAYLOAD} == hashes, 'Serialized payload changed')
        require(verified['contract'] == new_contract and verified['contract_sha256'] == jhash(new_contract),
                'Serialized contract mismatch')
        del verified
        results[filename] = dict(old_payload_hashes=hashes, new_payload_hashes=hashes,
                                old_checkpoint_sha256=sha(cp), new_checkpoint_sha256=sha(destination), counters=counters)
    # The driver is stopped. Every replacement has a preserved exact original.
    atomic_bytes(source / DRIVER, patched_driver)
    atomic_bytes(ROOT / 'inputs.json', new_bytes)
    for filename in results:
        os.replace(TRANSITION / filename, run / filename)
    new_resolved = dict(resolved, inputs=new, contract=new_contract, contract_sha256=jhash(new_contract))
    atomic_json(run / 'resolved.json', new_resolved)
    new_receipt = read(backup / 'run/checkpoint.json')
    new_receipt['bytes'] = (run / 'checkpoint-latest.pt').stat().st_size
    atomic_json(run / 'checkpoint.json', new_receipt)
    require(not torch.cuda.is_initialized(), 'Migration initialized CUDA')
    require(sha(ROOT / 'replay/index.json') == replay_index_sha, 'Replay index changed during migration')
    result = dict(ok=True, time=time.time(), purpose='eval25-to10', cpu_only=True,
                  old_inputs_sha256=old_contract['inputs_sha256'], new_inputs_sha256=sha(ROOT / 'inputs.json'),
                  old_contract=old_contract, new_contract=new_contract,
                  **results['checkpoint-latest.pt'], checkpoints=results,
                  training_payload_changed=[], save_policy_changed=False,
                  stop_truncated_episodes=0,
                  replay_path_identity_changed=False, source_sha256=new['port_source_manifest'])
    result['replay_index_sha256_before'] = replay_index_sha
    result['replay_index_sha256_after'] = sha(ROOT / 'replay/index.json')
    atomic_json(TRANSITION / 'migration-receipt.json', result)
    return result


def rollback():
    """Restore exact old authority while original owner is still SIGSTOP-held."""
    backup = TRANSITION / 'backup'
    require((TRANSITION / 'backup-ready.json').is_file(), 'Complete backup marker missing')
    for filename in ('checkpoint-latest.pt', 'checkpoint-last1.pt'):
        target = ROOT / 'run' / filename
        temp = target.with_name(filename + '.rollback-partial')
        os.link(backup / filename, temp); os.replace(temp, target)
    atomic_bytes(ROOT / 'source' / DRIVER, (backup / 'train_expo_formal.py').read_bytes())
    for name in ('inputs.json', 'run/resolved.json', 'run/checkpoint.json', 'current.json'):
        atomic_bytes(ROOT / name, (backup / name).read_bytes())
    atomic_json(TRANSITION / 'rollback.json', {'ok': True, 'time': time.time()})


if __name__ == '__main__':
    print(json.dumps(migrate()))
