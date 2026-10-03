"""Observe one WM run. No signals, CUDA initialization, or scheduling changes.

Use --once for a read-only stdout sample. Otherwise --output must name a new
JSONL file inside --run; collection ends when wm-exit.json appears.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def query(fields, kind='gpu'):
    result = subprocess.run(
        ['nvidia-smi', '--query-' + kind + '=' + fields,
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=8, check=True)
    return [[part.strip() for part in line.split(',')]
            for line in result.stdout.splitlines() if line.strip()]


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def process(anchor, boot, include_pss):
    try:
        pid = int(anchor['pid'])
        base = Path('/proc') / str(pid)
        status = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        if (base.stat().st_uid != os.getuid()
                or int(status[19]) != anchor.get('start', anchor.get('start_ticks'))
                or anchor.get('boot', boot) != boot or status[0] in ('Z', 'X')):
            return None
        values = {}
        for line in (base / 'status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'VmSwap'):
                values[key + '_kib'] = int(value.split()[0])
        if include_pss:
            for line in (base / 'smaps_rollup').read_text().splitlines():
                if line.startswith('Pss:'):
                    values['Pss_kib'] = int(line.split()[1])
        return {'pid': pid, 'start_ticks': int(status[19]),
                'state': status[0], **values}
    except (OSError, ValueError, KeyError):
        return None


def sample(run, gpus, include_pss):
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    row = {'time': time.time(), 'boot': boot, 'run': str(run),
           'terminal': read_json(run / 'wm-exit.json'), 'errors': []}
    row['host_kib'] = {
        line.split(':')[0]: int(line.split()[1])
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.split(':')[0] in ('MemAvailable', 'MemTotal', 'SwapFree', 'SwapTotal')}
    row['memory_pressure'] = Path('/proc/pressure/memory').read_text().strip()
    catalog = read_json(run / 'managed-identities.json', {})
    owned = {}
    for anchor in catalog.get('managed_processes', []):
        item = process(anchor, boot, include_pss)
        if item:
            owned[item['pid']] = item
    row['owned_processes'] = list(owned.values())
    try:
        rows = query('index,uuid,memory.total,memory.used,utilization.gpu')
        row['gpus'] = [dict(index=int(a[0]), uuid=a[1], total_mib=float(a[2]),
                            used_mib=float(a[3]), util_pct=float(a[4]))
                       for a in rows if int(a[0]) in gpus]
        selected = {a['uuid'] for a in row['gpus']}
        row['gpu_processes'] = [
            {'gpu_uuid': a[0], 'pid': int(a[1]), 'used_mib': a[2],
             'owned_identity_matches': int(a[1]) in owned}
            for a in query('gpu_uuid,pid,used_gpu_memory', 'compute-apps')
            if a[0] in selected]
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        row['errors'].append(type(error).__name__ + ': ' + str(error))
    try:
        with (run / 'metrics.log').open('rb') as stream:
            stream.seek(max(0, stream.seek(0, 2) - 14000))
            matches = re.findall(r'Global Step:\s*(\d+)', stream.read().decode(errors='replace'))
            row['last_completed_round'] = int(matches[-1]) if matches else None
    except OSError:
        row['last_completed_round'] = None
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    root = Path('/data/chenyiteng/projects/wan-goal-sz3/runs').resolve(strict=True)
    assert run.is_relative_to(root) and (run / 'launch.json').is_file()
    assert os.getuid() == 20001 and args.interval >= 5
    gpus = {int(x) for x in args.gpus.split(',')}
    assert gpus and gpus <= {4, 5, 6, 7}
    if args.once:
        print(json.dumps(sample(run, gpus, True)), flush=True)
        return
    assert args.output is not None
    output = args.output.resolve()
    assert output.parent == run and output.suffix == '.jsonl'
    started = time.monotonic()
    next_pss = started
    with output.open('x', encoding='utf-8', buffering=1) as stream:
        while True:
            now = time.monotonic()
            row = sample(run, gpus, now >= next_pss)
            if now >= next_pss:
                next_pss = now + 60
            stream.write(json.dumps(row) + '\n')
            if row['terminal'] or (args.max_seconds and now-started >= args.max_seconds):
                break
            time.sleep(max(0, args.interval - (time.monotonic()-now)))


if __name__ == '__main__':
    main()
