"""CPU-only reproduction of the deployed /proc identity monitor; own children only."""
import collections
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

R = Path('/data/chenyiteng/projects/wan-goal-sz3')
repaired = '--repaired' in sys.argv
BASE = (R/'scripts/resource_switch/common.py' if repaired else
        Path('/data/chenyiteng/projects/robodojo-openwam-sz3/scripts/wm-bridge-20261001-v5/common.py'))
assert os.getuid() == 20001
spec = importlib.util.spec_from_file_location('baseline_common', BASE)
baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
counts = collections.Counter(); examples = []; targets = set(); lock = threading.Lock(); stop = False

def watch():
    while not stop:
        with lock: pids = list(targets)
        for pid in pids:
            try:
                row = baseline.identity(pid); counts['identity_ok_' + row['state']] += 1
            except (FileNotFoundError, ProcessLookupError):
                counts['gone'] += 1
            except Exception as exc:
                counts[type(exc).__name__] += 1
                if len(examples) < 10:
                    sample = {'pid': pid, 'error': repr(exc), 'time': time.time()}
                    for filename in ('stat', 'status'):
                        try:
                            value = Path(f'/proc/{pid}/{filename}').read_text()
                            sample[filename] = value if filename == 'stat' else '\n'.join(
                                line for line in value.splitlines() if line.startswith(('Name:', 'State:', 'Uid:')))
                        except OSError as err: sample[filename] = repr(err)
                    try: sample['directory_uid'] = Path(f'/proc/{pid}').stat().st_uid
                    except OSError as err: sample['directory_uid'] = repr(err)
                    examples.append(sample)
        time.sleep(0.0001)

thread = threading.Thread(target=watch); thread.start()
started = time.monotonic()
try:
    for _ in range(2000):
        child = subprocess.Popen(['/bin/true'], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with lock: targets.add(child.pid)
        child.wait(timeout=5)
        with lock: targets.discard(child.pid)
finally:
    stop = True; thread.join(timeout=5)
assert not thread.is_alive()
report = {'time': time.time(), 'kernel': os.uname().release, 'uid': os.getuid(),
          'baseline': str(BASE), 'children': 2000, 'seconds': time.monotonic()-started,
          'counts': dict(counts), 'errors': examples, 'gpu_initialized': False,
          'scope': 'own /bin/true children; no unrelated signals or resource changes'}
path = R/'logs'/('process-identity-repaired-20261001.json' if repaired else 'process-identity-baseline-20261001.json')
with path.open('x') as stream: json.dump(report, stream, indent=2); stream.write('\n')
print(json.dumps(report), flush=True)
