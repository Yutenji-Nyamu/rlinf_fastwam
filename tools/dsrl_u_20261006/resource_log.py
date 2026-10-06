"""Bounded CPU resource evidence for the DSRL pair; never changes jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert os.getuid() == 1003 and not a.output.exists()
    started = time.time()
    with a.output.open('x', buffering=1) as stream:
        while time.time() - started < 7 * 86400:
            state = json.loads((a.control / 'status.json').read_text())
            xml = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], timeout=25))
            gpus = []
            pids = set()
            for index, node in enumerate(xml.findall('gpu')):
                processes = []
                for proc in node.findall('./processes/process_info'):
                    pid = int(proc.findtext('pid'))
                    processes.append({'pid': pid, 'type': proc.findtext('type')})
                    if index in (6, 7):
                        pids.add(pid)
                gpus.append({'index': index, 'uuid': node.findtext('uuid'),
                    'memory': node.findtext('./fb_memory_usage/used'),
                    'util': node.findtext('./utilization/gpu_util'), 'processes': processes,
                    'recovery': node.findtext('gpu_recovery_action')})
            resident = {}
            for pid in pids:
                proc = Path('/proc') / str(pid)
                try:
                    stat = proc.stat()
                    if stat.st_uid == 1003:
                        status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines() if ':' in line)
                        resident[pid] = {key: status.get(key, '').strip() for key in ('Name', 'VmRSS', 'VmHWM')}
                except FileNotFoundError:
                    pass
            mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            record = {'time': time.time(), 'heartbeat_age': time.time() - state['time'],
                'slots': {gpu: row['state'] for gpu, row in state['slots'].items()},
                'gpus': gpus, 'own_gpu_process_rss': resident, 'MemAvailable': mem['MemAvailable'].strip(),
                'free_bytes': {mount: shutil.disk_usage(mount).free for mount in ('/data', '/home')}}
            stream.write(json.dumps(record) + '\n')
            if (a.control / 'lease-returned.json').exists() or (a.control / 'owner-final.json').exists():
                break
            if record['heartbeat_age'] > 300:
                break
            time.sleep(10)


if __name__ == '__main__':
    main()
