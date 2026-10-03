"""Prepare/launch the reviewed same-cycle owner; preparation does not signal jobs.

`prepare` runs after a successful graphics probe and server CPU tests. `launch`
starts only the CPU owner, which independently revalidates every pinned input
before its identity-scoped transfer. Large checkpoints are neither read nor copied.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'launch'))
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--probe-proof', type=Path)
    parser.add_argument('--extra-pin', type=Path, action='append', default=[])
    args = parser.parse_args()
    stage = Path(os.path.abspath(args.stage))
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    spec = importlib.util.spec_from_file_location('_maintenance_owner', stage / 'maintenance_owner.py')
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    assert stage.resolve() == (owner.TRAIN.parent / 'gpu4567-fix-20261003').resolve()
    sys.path.insert(0, str(owner.SOURCE / 'tools'))
    from expo_smoke_owner import identity, owned, atomic
    contract_path = stage / 'snapshot-contract.json'
    if args.action == 'prepare':
        assert not contract_path.exists() and not (stage / 'owner-launched.json').exists()
        assert args.probe_proof is not None and args.probe_proof.is_file()
        # The root operator writes this small acceptance after reviewing the actual
        # renderer/reset/spawn probe evidence; all cited artifacts are then pinned.
        probe = read(args.probe_proof)
        assert probe.get('ok') is True and probe.get('physical_gpus') == [4, 5, 6, 7]
        tests = read(stage / 'cpu-tests.json')
        assert tests.get('ok') is True
        assert tests.get('maintenance_owner_sha256') == sha(stage / 'maintenance_owner.py')
        current = read(owner.CTRL / 'current.json')
        assert owned(current['owner']) and owned(current['child'])
        assert not any((owner.CTRL / name).exists() for name in owner.TERMINAL)
        assert not (owner.CYCLE / 'resumed-dispatched.json').exists()
        original_pins = read(owner.CTRL / 'launch-contract.json')['source_sha256']
        owner.pins_match(original_pins)
        files = dict(original_pins)
        extra = [stage / 'maintenance_owner.py', Path(__file__).resolve(), stage / 'bootstrap/sitecustomize.py',
                 stage / 'scope.json', stage / 'cpu-tests.json', args.probe_proof, *args.extra_pin]
        scope = read(stage / 'scope.json')
        extra.append(Path(scope['profile_path']))
        for path in extra:
            files[str(path)] = sha(path)
        for name, digest in read(owner.TRAIN / 'inputs.json')['port_source_manifest'].items():
            files[str(owner.SOURCE / name)] = digest
        queue_pins = {}
        queue_owners = {}
        for directory in owner.QUEUES:
            queue_pins[str(directory / 'plan.json')] = sha(directory / 'plan.json')
            saved = read(directory / 'owner-identity-continuation.json')
            live = identity(saved['pid'])
            assert owner.short_same(saved, live) and live['state'] not in ('Z', 'X')
            queue_owners[str(directory)] = live
            plan = read(directory / 'plan.json')
            for key in ('ops', 'queue_owner'):
                files[plan[key]] = sha(plan[key])
            files[str(directory / 'owner-identity-continuation.json')] = sha(directory / 'owner-identity-continuation.json')
        contract = dict(version=1, purpose='expo-gpu4567-same-cycle', time=time.time(),
                        stage=str(stage), control=str(owner.CTRL), train=str(owner.TRAIN), cycle=str(owner.CYCLE),
                        old_current=current, files_sha256=files,
                        inputs_sha256=sha(owner.TRAIN / 'inputs.json'),
                        cycle_plan_sha256=sha(owner.CYCLE / 'plan.json'),
                        queue_plan_sha256=queue_pins, queue_owners=queue_owners,
                        graphics_probe_verified=True, graphics_probe_receipt=str(args.probe_proof))
        owner.validate_contract(contract, stage)
        owner.pins_match(files)
        atomic(contract_path, contract)
        print(json.dumps(dict(ok=True, action='prepared', contract=str(contract_path),
                              contract_sha256=sha(contract_path), old_owner=current['owner'], child=current['child'])))
        return
    contract = read(contract_path)
    owner.validate_contract(contract, stage)
    owner.pins_match(contract['files_sha256'])
    assert not (stage / 'owner-launched.json').exists() and not (stage / 'handoff-intent.json').exists()
    assert owned(contract['old_current']['owner']) and owned(contract['old_current']['child'])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONFAULTHANDLER='1')
    environment.pop('RLINF_EXPO_GPU_SCOPE_MANIFEST', None)
    # The maintenance interpreter must not load GPU bootstrap from its parent's PYTHONPATH.
    environment['PYTHONPATH'] = str(owner.SOURCE / 'tools')
    command = [owner.RLT_PY, '-X', 'faulthandler', '-u', '-B', str(stage / 'maintenance_owner.py'), '--stage', str(stage)]
    with (stage / 'owner.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=stage, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launched = dict(time=time.time(), owner=identity(child.pid), command=command, contract_sha256=sha(contract_path))
    atomic(stage / 'owner-launched.json', launched)
    print(json.dumps(launched))


if __name__ == '__main__':
    main()
