"""One-shot, CPU-only migration of the zero-action turn-switch EXPO checkpoint.

Run only on SZ2 after reviewed new inputs/source pins exist. The old scope is
read-only. Only outer run/input identity and replay directory identity change;
the latter is required by FormalReplay's strict root/inode restore contract.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

OLD = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-20261001')
NEW = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002')
PAYLOAD = {'base', 'core', 'replay', 'cadence', 'rng', 'progress'}
CHANGED_SOURCE = {'examples/embodiment/train_expo_formal.py',
                  'rlinf/algorithms/expo_ft/formal_eval.py'}
HELPER = 'rlinf/algorithms/expo_ft/lifecycle.py'
CONTROL_SOURCE = {
    'tools/expo_formal_owner.py', 'tools/expo_formal_resources.py',
    'tools/launch_expo_repair.py', 'tools/expo_rlt_switch.py',
    'tools/expo_smoke_owner.py', 'tools/expo_process.py',
    'tests/native_expo_lifecycle.py', 'tests/test_expo_lifecycle.py',
    'tests/test_expo_repair_owner.py',
}
REVIEWED_SOURCE = CHANGED_SOURCE | {HELPER} | CONTROL_SOURCE


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path); before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        require((os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino)
                == (before.st_dev, before.st_ino), 'File replaced before SHA: ' + str(path))
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'File changed during SHA: ' + str(path))
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_contract(root, inputs_sha256):
    """Match the driver's Path.resolve() contract, including mount aliases."""
    root = Path(root)
    return dict(version=4, run=str((root / 'run').resolve()),
                inputs_path=str((root / 'inputs.json').resolve()),
                inputs_sha256=inputs_sha256, max_physical_actions=20000,
                evaluation_enabled=True)


def read(path):
    return json.loads(Path(path).read_text())


def owned(path):
    path = Path(path)
    require(path.exists() and not path.is_symlink() and path.stat().st_uid == os.getuid(),
            'Missing, symlinked, or foreign-owned path: ' + str(path))
    return path


def write_bytes_once(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def write_json_once(path, value):
    write_bytes_once(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def check_inputs(old, new, old_root=OLD, new_root=NEW):
    """Only source pins and two training-environment output routes may differ."""
    expected = copy.deepcopy(old)
    old_manifest = expected.pop('port_source_manifest')
    candidate = copy.deepcopy(new); new_manifest = candidate.pop('port_source_manifest')
    for parent, key in (('task_config', 'save_path'), ('video_cfg', 'video_base_dir')):
        value = expected['env'][parent][key]
        require(value.startswith(str(old_root) + '/'), 'Old env output is outside old scope')
        expected['env'][parent][key] = str(new_root) + value[len(str(old_root)):]
    require(candidate == expected, 'Inputs differ outside source pins and approved output routes')
    require(set(new_manifest) == set(old_manifest) | {HELPER} | CONTROL_SOURCE,
            'Added/removed source is outside the exact lifecycle/control manifest')
    changed = {key for key in old_manifest if old_manifest[key] != new_manifest[key]}
    require(changed <= CHANGED_SOURCE | CONTROL_SOURCE,
            'Algorithm/backend/replay/cadence or another protected source changed: ' + repr(changed))
    require(CHANGED_SOURCE <= changed, 'Both formal driver and eval must carry reviewed lifecycle fix')
    return old_manifest, new_manifest


def check_source(root, manifest):
    source = owned(root / 'source').resolve()
    for name, expected in manifest.items():
        path = source / name
        require(path.resolve().is_relative_to(source), 'Source manifest escaped root')
        owned(path)
        require(sha(path) == expected, 'Source SHA mismatch: ' + name)


def load_digest(torch):
    """Extract the unchanged actual digest without importing GPU/model modules."""
    source = OLD / 'source/examples/embodiment/train_expo_ft.py'
    tree = ast.parse(source.read_text())
    node, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'digest']
    namespace = {'torch': torch, 'hashlib': hashlib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['digest']


def check_finite(value, torch, np):
    if torch.is_tensor(value):
        require(value.device.type == 'cpu', 'Migration loaded a non-CPU tensor')
        if value.is_floating_point() or value.is_complex():
            require(bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor')
    elif isinstance(value, dict):
        for item in value.values(): check_finite(item, torch, np)
    elif isinstance(value, (tuple, list)):
        for item in value: check_finite(item, torch, np)
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.inexact):
        require(bool(np.isfinite(value).all()), 'Nonfinite checkpoint array')
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(value), 'Nonfinite checkpoint scalar')


def check_zero_state(saved, receipt, index):
    require(receipt['reason'] == 'fresh-base', 'Only a fresh-base checkpoint is eligible')
    require(set(saved['hashes']) == PAYLOAD, 'Checkpoint payload schema differs')
    counters = saved['cadence']['counters']
    require(set(counters) == {'episodes_completed', 'physical_actions', 'warmup_actions',
            'post_warmup_actions', 'carry_actions', 'pending_calls', 'completed_calls',
            'budget_truncated_episodes'}, 'Cadence counter schema differs')
    require(all(type(value) is int and value == 0 for value in counters.values()),
            'Cadence is not an untouched zero-action state')
    require(saved['base']['base_updates'] == 0, 'Base already learned')
    extra = saved['core']['_extra_state']
    require(all(extra[key] == 0 for key in ('update_calls', 'critic_steps', 'editor_steps', 'temperature_steps')),
            'Learner already updated')
    replay = saved['replay']; progress = saved['progress']
    require(replay['online_entries'] == [] and index['online_entries'] == [], 'Online replay is nonempty')
    require(replay['samples_q'] == replay['samples_fm'] == 0, 'Replay was already sampled')
    require(progress['evaluation_initial'] is False and progress['evaluation_final'] is False
            and progress['evaluation_periodic'] == [] and progress['evaluation_summaries'] == {}
            and progress['online_success'] == progress['stopped_episodes'] == progress['horizon_term_precedence'] == 0
            and progress['training_seed_sha256'] is None, 'Progress is not original fresh-base state')


def rebind_replay(replay, new_root, new_identity, digest):
    migrated = copy.deepcopy(replay)
    migrated['root'] = str(new_root)
    migrated['root_identity'] = list(new_identity)
    before = {key: digest(value) for key, value in replay.items()}
    after = {key: digest(value) for key, value in migrated.items()}
    changed = {key for key in before if before[key] != after[key]}
    require(changed == {'root', 'root_identity'}, 'Replay changed beyond exact path/inode migration')
    return migrated, {'changed_keys': sorted(changed), 'before': before, 'after': after}


def check_eval(result, inputs):
    evaluation = inputs['evaluation']; cfg = copy.deepcopy(evaluation['config'])
    expected_sha = cfg.pop('_expo_seed_sha256')
    require(sha(owned(evaluation['seed_path'])) == expected_sha, 'Fixed eval seed SHA differs')
    seeds = read(evaluation['seed_path'])[cfg['task_config']['task_name']]['success_seeds']
    expected = dict(task=cfg['task_config']['task_name'], config_sha256=json_hash(cfg),
                    seed_sha256=expected_sha, seeds=seeds, policy='raw-base', base_updates=0,
                    learner_calls=0, steps_per_episode=200, parallel_envs=4, n_base=1, eval_noise_seed=90042)
    require(result['ok'] is True and result['contract'] == expected and result['env_closed'] is True,
            'Original initial evaluation contract differs')
    require(len(seeds) == 20 and len(set(seeds)) == 20 and result['episodes'] == 20
            and result['evaluation_physical_actions'] == 4000 and result['training_budgeted_actions'] == 0,
            'Initial evaluation budget differs')
    require([row['seed'] for row in result['rows']] == seeds
            and all(row['physical_actions'] == 200 and type(row['success_once']) is bool for row in result['rows']),
            'Initial evaluation per-seed records differ')
    successes = sum(row['success_once'] for row in result['rows'])
    require(result['successes'] == successes and result['success_rate'] == successes / 20,
            'Initial evaluation aggregate differs from rows')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviewed-runtime-pins', required=True,
                        help='JSON mapping of every exact REVIEWED_SOURCE lifecycle/control file to SHA256')
    args = parser.parse_args()
    require(os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02', 'This migration is SZ2-only')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.pop('RLINF_EXPO_SAPIEN_RENDER_DEVICE', None)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    require(not torch.cuda.is_initialized(), 'CUDA already initialized')
    for root in (OLD, NEW): owned(root)
    old_current = read(owned(OLD / 'current.json')); old_final = read(owned(OLD / 'final.json'))
    require(old_current['status'] == 'RLT_RESTORED' and old_final['gpu_released'] is True
            and old_final['rlt_first_rounds_verified'] is True, 'Old resource cycle has not finished')
    for identity in (old_current['owner'], old_current['child']):
        proc = Path('/proc') / str(identity['pid'])
        if proc.exists():
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            require(int(fields[19]) != identity['start_ticks'] or fields[0] in ('Z', 'X'),
                    'Original EXPO owner/driver is still alive')
    old_inputs_path = owned(OLD / 'inputs.json'); new_inputs_path = owned(NEW / 'inputs.json')
    old_inputs, new_inputs = read(old_inputs_path), read(new_inputs_path)
    old_pins, new_pins = check_inputs(old_inputs, new_inputs)
    reviewed = read(owned(args.reviewed_runtime_pins))
    require(set(reviewed) == REVIEWED_SOURCE
            and reviewed == {name: new_pins[name] for name in reviewed}, 'Runtime review pins differ')
    check_source(OLD, old_pins); check_source(NEW, new_pins)
    old_run = OLD / 'run'; new_run = NEW / 'run'; old_replay = owned(OLD / 'replay')
    old_cp = owned(old_run / 'checkpoint-latest.pt'); old_cp_sha = sha(old_cp)
    resolved = read(owned(old_run / 'resolved.json')); receipt = read(owned(old_run / 'checkpoint.json'))
    old_contract = make_contract(OLD, sha(old_inputs_path))
    require(resolved['contract'] == old_contract and resolved['contract_sha256'] == json_hash(old_contract)
            and resolved['inputs'] == old_inputs, 'Old resolved contract/input authority differs')
    digest = load_digest(torch)
    saved = torch.load(old_cp, map_location='cpu', weights_only=False)
    require(set(saved) == PAYLOAD | {'version', 'contract', 'contract_sha256', 'hashes'}
            and saved['version'] == 4 and saved['contract'] == old_contract
            and saved['contract_sha256'] == json_hash(old_contract), 'Old checkpoint contract differs')
    check_finite(saved, torch, np)
    original_hashes = {key: digest(saved[key]) for key in PAYLOAD}
    require(original_hashes == saved['hashes'] == receipt['hashes'], 'Old payload hashes differ')
    require(receipt['path'] == str(old_cp.resolve()) and receipt['bytes'] == old_cp.stat().st_size
            and receipt['finite'] is True and receipt['cadence'] == saved['cadence']
            and receipt['core_updates'] == receipt['base_updates'] == 0, 'Checkpoint receipt differs')
    index_path = owned(old_replay / 'index.json'); index_bytes = index_path.read_bytes(); index = json.loads(index_bytes)
    check_zero_state(saved, receipt, index)
    require(saved['replay']['root'] == str(old_replay.resolve())
            and saved['replay']['root_identity'] == [old_replay.stat().st_dev, old_replay.stat().st_ino, os.getuid()]
            and saved['replay']['root_id'] == index['root_id']
            and saved['replay']['version'] == index['version']
            and saved['replay']['contract_sha256'] == index['contract_sha256'] == json_hash(saved['replay']['contract']),
            'Old replay identity/index differs')
    require(not any(owned(old_replay / 'online').iterdir()), 'Old online directory is not empty')
    eval_path = owned(old_run / 'evaluations/initial-base/complete.json'); eval_bytes = eval_path.read_bytes()
    check_eval(json.loads(eval_bytes), old_inputs); check_eval(json.loads(eval_bytes), new_inputs)
    new_contract = make_contract(NEW, sha(new_inputs_path))
    new_replay = NEW / 'replay'
    require(not new_run.exists() and not new_replay.exists(), 'Migration output already exists; inspect receipts')
    new_run.mkdir(mode=0o700); new_replay.mkdir(mode=0o700); (new_replay / 'online').mkdir(mode=0o700)
    write_bytes_once(new_replay / 'index.json', index_bytes)
    saved['replay'], replay_diff = rebind_replay(saved['replay'], new_replay.resolve(),
        [new_replay.stat().st_dev, new_replay.stat().st_ino, os.getuid()], digest)
    saved['contract'] = new_contract; saved['contract_sha256'] = json_hash(new_contract)
    saved['hashes'] = {key: digest(saved[key]) for key in PAYLOAD}
    require({key for key in PAYLOAD if original_hashes[key] != saved['hashes'][key]} == {'replay'},
            'A non-replay payload changed during migration')
    # Exercise the actual unchanged replay restore, dataset pins and physical-window counts.
    sys.path.insert(0, str(NEW / 'source'))
    from rlinf.algorithms.expo_ft.formal_replay import FormalReplay
    replay = FormalReplay(new_replay, new_inputs['demo_path'], seed=new_inputs['formal']['seed'])
    replay.load_state_dict(saved['replay'])
    require(digest(replay.state_dict()) == saved['hashes']['replay'], 'Actual replay round-trip differs')
    new_cp = new_run / 'checkpoint-latest.pt'; temporary = new_run / 'checkpoint-latest.pt.partial'
    with temporary.open('xb') as stream:
        torch.save(saved, stream); stream.flush(); os.fsync(stream.fileno())
    os.rename(temporary, new_cp)
    verified = torch.load(new_cp, map_location='cpu', weights_only=False)
    check_finite(verified, torch, np)
    require({key: digest(verified[key]) for key in PAYLOAD} == saved['hashes'], 'New serialized payload differs')
    require(verified['contract'] == new_contract and verified['contract_sha256'] == json_hash(new_contract),
            'New serialized outer contract differs')
    write_json_once(new_run / 'resolved.json', dict(inputs=new_inputs,
        cli=dict(inputs=str(new_inputs_path), run=str(new_run), max_physical_actions=20000,
                 resume=str(new_cp), enable_evaluation=True), contract=new_contract, contract_sha256=json_hash(new_contract)))
    new_receipt = dict(receipt, path=str(new_cp.resolve()), bytes=new_cp.stat().st_size, hashes=saved['hashes'])
    write_json_once(new_run / 'checkpoint.json', new_receipt)
    new_eval = new_run / 'evaluations/initial-base'; new_eval.mkdir(parents=True, mode=0o700)
    write_bytes_once(new_eval / 'complete.json', eval_bytes)
    require(sha(old_cp) == old_cp_sha and index_path.read_bytes() == index_bytes
            and eval_path.read_bytes() == eval_bytes, 'Old authority changed during migration')
    require(not torch.cuda.is_initialized(), 'CPU-only migration initialized CUDA')
    result = dict(ok=True, cpu_only=True, cuda_initialized=False,
        old_checkpoint=str(old_cp), new_checkpoint=str(new_cp), old_checkpoint_sha256=old_cp_sha,
        new_checkpoint_sha256=sha(new_cp), old_payload_hashes=original_hashes, new_payload_hashes=saved['hashes'],
        payload_changed=['replay'], replay_field_diff=replay_diff,
        old_contract=old_contract, new_contract=new_contract, reviewed_runtime_pins=reviewed,
        initial_evaluation_sha256=hashlib.sha256(eval_bytes).hexdigest(), initial_progress_unchanged=True,
        replay_index_byte_identical=True, actual_replay_restore_verified=True, original_checkpoint_preserved=True)
    write_json_once(NEW / 'checkpoint-migration.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
