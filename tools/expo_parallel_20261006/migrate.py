"""Migrate only device metadata after a successful isolated capacity test."""
import argparse
import copy
import os
import sys
import time
from common import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--devices', type=int, choices=(1, 2), required=True)
    a = p.parse_args()
    assert os.getuid() == 20001 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert read(STAGE / f'trial-{a.devices}/result.json')['ok']
    assert not (STAGE / 'migration.json').exists()
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(SOURCE / 'examples/embodiment'))
    from train_expo_ft import digest
    run = TRAIN / 'run'; backup = STAGE / 'baseline'
    old = read(backup / 'inputs.json'); new = copy.deepcopy(old)
    for k in ('formal', 'core'): new[k]['parallel_devices'] = a.devices
    name = 'examples/embodiment/train_expo_formal.py'
    new['port_source_manifest'][name] = sha(SOURCE / name)
    # Validate the exact leaf whitelist before replacing production metadata.
    expected = copy.deepcopy(new)
    for k in ('formal', 'core'): expected[k]['parallel_devices'] = 4
    expected['port_source_manifest'][name] = old['port_source_manifest'][name]
    assert expected == old
    resolved = read(backup / 'resolved.json'); old_contract = resolved['contract']
    assert sha(TRAIN / 'inputs.json') == old_contract['inputs_sha256']
    new_bytes = (json.dumps(new, indent=2) + '\n').encode()
    contract = dict(old_contract, inputs_sha256=hashlib.sha256(new_bytes).hexdigest())
    result = {}
    for filename in ('checkpoint-latest.pt', 'checkpoint-last1.pt'):
        saved = torch.load(backup / filename, map_location='cpu', weights_only=False)
        before = {k: digest(saved[k]) for k in PAYLOAD}
        assert before == saved['hashes'] and saved['contract'] == old_contract
        # Hash all learning state independently from the three changed metadata fields.
        def invariant(s):
            base = dict(s['base']); base.pop('contract'); base.pop('contract_sha256')
            core = dict(s['core']); extra = dict(core['_extra_state']); extra.pop('config'); core['_extra_state'] = extra
            rng = dict(s['rng']); rng.pop('cuda')
            return digest((base, core, s['replay'], s['cadence'], s['progress'], rng))
        fixed = invariant(saved); cuda_before = [digest(v) for v in saved['rng']['cuda']]
        change_devices(saved, a.devices)
        assert invariant(saved) == fixed
        assert [digest(v) for v in saved['rng']['cuda']] == cuda_before[:a.devices]
        after = {k: digest(saved[k]) for k in PAYLOAD}
        saved.update(contract=contract, contract_sha256=jhash(contract), hashes=after)
        target = STAGE / filename
        with target.open('xb') as f:
            torch.save(saved, f); f.flush(); os.fsync(f.fileno())
        del saved
        check = torch.load(target, map_location='cpu', weights_only=False)
        assert {k: digest(check[k]) for k in PAYLOAD} == after
        result[filename] = dict(before=before, after=after, unchanged_learning_state=True,
                                cadence=check['cadence'], bytes=target.stat().st_size)
        del check
    assert sha(TRAIN / 'replay/index.json') == read(STAGE / 'paused.json')['replay_index_sha256']
    # All files are complete before the sole stopped owner commits this migration.
    (TRAIN / 'inputs.json.parallel-partial').write_bytes(new_bytes)
    (TRAIN / 'inputs.json.parallel-partial').replace(TRAIN / 'inputs.json')
    for filename in result: (STAGE / filename).replace(run / filename)
    atomic(run / 'resolved.json', dict(resolved, inputs=new, contract=contract, contract_sha256=jhash(contract)))
    cp = read(backup / 'checkpoint.json')
    cp.update(hashes=result['checkpoint-latest.pt']['after'], bytes=result['checkpoint-latest.pt']['bytes'],
              reason=f'authorized-device-migration-4-to-{a.devices}')
    atomic(run / 'checkpoint.json', cp)
    atomic(STAGE / 'migration.json', dict(ok=True, devices=a.devices, time=time.time(), checkpoint_results=result,
           inputs_sha256=sha(TRAIN / 'inputs.json'), cuda_rng_mapping=list(range(a.devices)),
           method_and_counters_unchanged=True, backup=str(backup), replay_unchanged=True))
    print(json.dumps({'ok': True, 'devices': a.devices}))

if __name__ == '__main__': main()
