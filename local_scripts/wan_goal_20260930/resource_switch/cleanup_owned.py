"""Stop only this WM token/catalog. Never invoke global Ray lifecycle commands."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from common import UID, account, alive, atomic, identity, own_path, read
from wm_stage import Catalog, TAG


def gpu_processes():
    xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=30)
    tree = ET.fromstring(xml); result = []
    assert len(tree.findall('gpu')) >= 8, 'Unexpected GPU inventory'
    for index, gpu in enumerate(tree.findall('gpu')):
        if index not in (4, 5, 6, 7): continue
        for entry in gpu.findall('./processes/process_info'):
            pid = entry.findtext('pid', '')
            if pid.isdigit(): result.append(dict(gpu=index, uuid=gpu.findtext('uuid'), pid=int(pid)))
    return result


def main():
    account()
    token = os.environ[TAG]
    run = own_path(os.environ['WAN_GOAL_RUN_DIR'])
    owner = read(run / 'owner.json')
    assert owner['owner_token'] == token and owner['cycle_id'] == os.environ['WAN_GOAL_CYCLE_ID']
    identities = own_path(os.environ['WAN_GOAL_IDENTITIES'])
    assert identities == run / 'managed-identities.json'
    target = own_path(os.environ['WAN_GOAL_CLEANUP_RECEIPT'], exists=False)
    assert target == run / 'wm-cleanup.json' and not target.exists()
    ray = owner['ray']
    assert ray['address'] == os.environ['RAY_ADDRESS']
    assert ray['namespace'] == os.environ['CLUSTER_NAMESPACE']
    assert ray['temp_dir'] == os.environ['WAN_GOAL_RAY_TMPDIR']
    recorded = read(identities)
    assert recorded['owner_token'] == token
    catalog = Catalog(run, token)
    for row in recorded['managed_processes']:
        assert row['uid'] == UID and isinstance(row['boot'], str)
        if alive(row): catalog.add(row, 'persisted-owner-catalog')
    mine = identity(os.getpid()); actions = []

    def targets():
        catalog.scan()
        # The callback itself is token-tagged. Its synchronous probe children
        # must also survive until they return; they are not workload targets.
        excluded = {mine['pid']}
        while True:
            added = {r['pid'] for r in catalog.rows.values() if r['ppid'] in excluded} - excluded
            if not added: break
            excluded |= added
        return [r for r in catalog.rows.values() if r['pid'] not in excluded and alive(r)]

    for sig, seconds in ((signal.SIGTERM, 30), (signal.SIGKILL, 20)):
        deadline = time.monotonic() + seconds
        signaled = set()
        while time.monotonic() < deadline:
            live = targets()
            if not live: break
            for row in live:
                key = (row['pid'], row['start'], row['boot'])
                if key in signaled: continue
                try: current = identity(row['pid'])
                except (FileNotFoundError, ProcessLookupError): continue
                catalog.signal(current, sig)
                actions.append(dict(time=time.time(), signal=sig.name, identity=current))
                signaled.add(key)
            atomic(run / 'cleanup-actions.json', dict(owner_token=token, actions=actions))
            time.sleep(1)
    remaining = targets()
    assert not remaining, 'Owned WM processes remain: ' + json.dumps(remaining)
    # Require a short stable empty interval to catch delayed owned children.
    for _ in range(3):
        time.sleep(1)
        assert not targets(), 'A late WM child appeared'
    assert not gpu_processes(), 'GPU 4–7 have remaining contexts; no unrelated process is signaled'
    receipt = dict(time=time.time(), cycle_id=owner['cycle_id'], owner_token=token,
                   physical_gpus=[4, 5, 6, 7], processes_clear=True, gpus_released=True,
                   ray=dict(address=ray['address'], namespace=ray['namespace'],
                            temp_dir=ray['temp_dir'], stopped=True),
                   managed_processes=list(catalog.rows.values()), actions=actions,
                   callback_identity=mine, verification='owned token/catalog empty; physical GPU contexts empty')
    atomic(target, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
