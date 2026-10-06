"""Read-only compact status for the original-pi0.5 two-card formal run."""
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import urllib.request

assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
F = S / 'lift-two-gpu-from0-v2'
O = S / 'runs/lift-two-gpu-from0-v2'
read = lambda p: json.loads(Path(p).read_text())


def tail(p, n=4000):
    if not p.is_file():
        return None
    with p.open('rb') as f:
        f.seek(max(0, p.stat().st_size - n))
        return f.read().decode(errors='replace')


def alive(identity):
    p = Path('/proc') / str(identity['pid'])
    if not p.exists():
        return False
    stat = (p / 'stat').read_text().split(') ', 1)[1].split()
    return p.stat().st_uid == 20001 and int(stat[19]) == identity['start'] and stat[0] not in ['Z', 'X']


out = {'time': datetime.datetime.now().astimezone().isoformat(), 'files': {}, 'rlt': {}, 'logs': {}}
for name in ['owner-identity.json', 'state.json', 'error.json', 'final.json', 'recovery-error.json',
             'formal/result.json', 'formal/verified-placement.json', 'formal/driver-identity.json',
             'services/wm5/service-cpu-ready.json']:
    p = O / name
    if p.is_file():
        out['files'][name] = read(p)
        if name.endswith('identity.json'):
            out['files'][name]['alive_now'] = alive(out['files'][name])
for name in ['ready.json', 'launch.json', 'resource-timeout-cpu-test.json']:
    p = F / name
    if p.is_file():
        out['files'][name] = read(p)
for p in [F / 'preparation.log', F / 'owner-console.log', O / 'formal/driver.log', O / 'services/wm5/service.log']:
    text = tail(p)
    if text is not None:
        out['logs'][str(p.relative_to(S))] = text
cfg_path = F / 'prepared/formal.yaml'
if cfg_path.is_file():
    cfg = read(cfg_path)
    out['config'] = {'resume_dir': cfg['runner']['resume_dir'], 'ckpt_path': cfg['runner'].get('ckpt_path'),
                     'max_steps': cfg['runner']['max_steps'], 'placement': cfg['cluster']['component_placement'],
                     'actor_model': cfg['actor']['model'], 'N': cfg['env']['train']['total_num_envs'],
                     'R': cfg['env']['train']['rollout_epoch'], 'G': cfg['algorithm']['group_size'],
                     'actor_micro': cfg['actor']['micro_batch_size'], 'global_batch': cfg['actor']['global_batch_size'],
                     'resume_source_world_size': cfg['actor']['fsdp_config']['resume_source_world_size']}
    out['worker_tails'] = {}
    for p in (O / 'formal').rglob('*'):
        if p.is_file() and p.suffix in ['.log', '.out'] and p.name != 'driver.log':
            out['worker_tails'][str(p.relative_to(O))] = tail(p, 6000)
try:
    with urllib.request.urlopen('http://127.0.0.1:18985/health', timeout=3) as response:
        health = json.load(response)
    out['health'] = {k: v for k, v in health.items() if k in ['ok', 'pid', 'physical_gpu', 'is_offloaded', 'wm_batch_size']}
except (OSError, ValueError) as exc:
    out['health'] = {'unavailable': type(exc).__name__}
oldcp = read(S / 'lift-pot-v1/prepared-cycles/cycle/plan.json')
for key in ['gpu6', 'gpu7']:
    identity = read(Path(oldcp['runs'][key]['new_run']) / 'runtime/driver-identity.json')
    out['rlt'][key] = {'pid': identity['pid'], 'start': identity['start'], 'alive': alive(identity)}
cp_path = F / 'cycles/cycle/plan.json'
if cp_path.is_file():
    cp = read(cp_path)
    out['rlt_borrowed'] = (cp_path.parent / 'rlt-stopped.json').is_file()
    out['rlt_queued'] = {k: {'gpus': r['gpus'], 'new_run': r['new_run'],
                            'checkpoint': r['recovery']['checkpoint']['step']} for k, r in cp['runs'].items()}
events = O / 'services/wm5/records/service-events.jsonl'
if events.is_file():
    rows = [json.loads(x) for x in events.read_text().splitlines()]
    completed = [r for r in rows if r.get('event') == 'batch_completed']
    out['wm_batches'] = {'count': len(completed), 'recent': completed[-2:]}
for key, argv in [('gpu', ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits']),
                  ('contexts', ['nvidia-smi', 'pmon', '-c', '1', '-s', 'm'])]:
    try:
        out[key] = subprocess.check_output(argv, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        out[key] = 'read-only query timeout'
print(json.dumps(out))
