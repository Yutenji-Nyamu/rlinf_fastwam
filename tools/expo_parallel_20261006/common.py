"""Fixed scope for the authorized 1 -> 2 GPU trial of the existing 60k run."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001')
TRAIN = ROOT / 'formal-turn-switch-repair-20261002'
OLD = ROOT / 'continue-60k-20261005'
STAGE = ROOT / 'parallel-trial-20261006'
SOURCE = STAGE / 'source'
TOOLS = SOURCE / 'tools/expo_parallel_20261006'
CYCLE = OLD / 'rlt-cycle-expo60k-20261005-v1'
EXPO_PY = '/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python'
CPU_PY = '/usr/bin/python3'
RLT_PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
QUEUES = [Path('/data/chenyiteng/deployment-20261002/rlt-next6-' + task)
          for task in ('place_object_stand', 'move_playingcard_away')]
UUIDS = ['GPU-a0a252d6-828d-29e1-1fd2-65187f573f4d', 'GPU-2cd891ea-180d-da39-6419-2d7033f8b21b',
         'GPU-dc5d6921-fa81-b666-bac7-566c126f1dd4', 'GPU-3c6321c1-3e58-c071-3867-533391152fe7']
PAYLOAD = ('base', 'core', 'replay', 'cadence', 'rng', 'progress')

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

def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial-' + str(os.getpid()))
    with temp.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def change_devices(saved, devices):
    """Change placement metadata/RNG device list only, without touching tensors."""
    import copy
    if type(devices) is not int or devices not in (1, 2, 4):
        raise ValueError('Invalid trial device count')
    old_devices = saved['base']['contract']['parallel']['devices']
    assert old_devices == 4
    assert saved['core']['_extra_state']['config']['parallel_devices'] == old_devices
    assert len(saved['rng']['cuda']) == old_devices
    saved['base']['contract'] = copy.deepcopy(saved['base']['contract'])
    saved['base']['contract']['parallel']['devices'] = devices
    saved['base']['contract_sha256'] = jhash(saved['base']['contract'])
    saved['core']['_extra_state']['config'] = dict(saved['core']['_extra_state']['config'], parallel_devices=devices)
    saved['rng']['cuda'] = saved['rng']['cuda'][:devices]
    return saved
