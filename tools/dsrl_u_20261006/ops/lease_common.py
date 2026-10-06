"""CPU-only helpers for the exclusive DSRL6/7 lease."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time

MASKS = ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, exclusive=False):
    path = Path(path)
    temporary = path if exclusive else path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        temporary.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked(control):
    control = Path(control)
    p = read(control / 'plan.json')
    assert read(control / 'prepared.json')['plan_sha256'] == sha(control / 'plan.json')
    assert os.getuid() == p['uid'] == 1003 and socket.gethostname() == p['hostname'] == 'admin'
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == p['boot_id']
    assert set(p['slots']) == {'6', '7'}
    for path, digest in p['pins'].items():
        assert sha(path) == digest, 'Pinned lease input changed: ' + path
    b = load(p['common_source'], 'dsrl_trusted_common')
    op = load(p['rlt_ops'], 'dsrl_trusted_ops')
    return p, b, op


def original_protected(p, b):
    # Natural completion is allowed; a different driver at the same run is not.
    for row in p['protected'].values():
        live = b.proc(row['driver']['pid'])
        if live and live['state'] not in ('Z', 'X'):
            assert all(live[k] == row['driver'][k] for k in ('pid', 'uid', 'start'))
        actual = read(Path(row['run']) / 'runtime/driver-identity.json')
        assert all(actual[k] == row['driver'][k] for k in ('pid', 'uid', 'start'))
    actual = read(Path(p['original_stage']) / 'queue-status.json')['owner']
    assert all(actual[k] == p['original_owner'][k] for k in ('pid', 'uid', 'start'))


def environment(p, gpu, *, probe=False):
    # A probe's mask is constructed per child and never touches owner state.
    env = os.environ.copy()
    for key in MASKS + ('LD_PRELOAD', 'RLINF_OPENDW_GPU_SCOPE_MANIFEST'):
        env.pop(key, None)
    env.update(p['slots'][str(gpu)]['scope_env'])
    env.update(HOME='/home/chenyiteng', PYTHONDONTWRITEBYTECODE='1')
    if probe:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def gpu_released(p, b, op, gpu, namespace=None):
    row = b.gpu_snapshot()[int(gpu)]
    actors = op.active_actors(p, namespace) if namespace else []
    op.validate_scoped_actors(p, actors)
    return (not actors and b.empty_target(row, p['slots'][str(gpu)]['gpu_uuid'])), row, actors
