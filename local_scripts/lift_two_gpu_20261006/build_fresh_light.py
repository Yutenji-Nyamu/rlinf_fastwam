"""Extract fresh-run receipts without publishing environments or raw logs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003')
value = json.loads((OUT / (sys.argv[1] + '.out')).read_text())
cfg = value['config']
model = cfg.pop('actor_model')
cfg['initial_model'] = {k: v for k, v in model.items() if k in ['model_type', 'model_path', 'model_name', 'pretrained_path']}
wm = value.get('wm_batches', {})
recent = [{k: r[k] for k in ['timestamp_utc', 'actual_wm_batch', 'seconds', 'world_model_seconds',
                            'reward_seconds', 'wm_peak', 'outputs_finite'] if k in r} for r in wm.get('recent', [])]
proofs = []
for source, text in value.get('worker_tails', {}).items():
    for line in text.splitlines():
        if any(x in line for x in ['WMRL_POLICY_ACTUAL_BATCH ', 'WMRL_ACTOR_MEMORY ', 'Traceback (most recent',
                                   'OutOfMemoryError', 'Epoch: ', 'Epoch ', 'epoch: ', 'Global step']):
            proofs.append({'source': source, 'line': line[:1000]})
light = {'observed_at': value['time'], 'task': 'lift_pot', 'third_task': True,
         'start_policy': 'original_pi05', 'rl_start_step': 0, 'optimizer_resumed': False,
         'source_head': '9ce50c602c5e773e87e3306218fdde0cf167e8a8',
         'owner_directory': '/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/runs/lift-two-gpu-from0-v2',
         'config': cfg, 'receipts': value['files'], 'health': value['health'],
         'rlt_gpu6_gpu7': value['rlt'], 'rlt_gpu4_gpu5_borrowed': value.get('rlt_borrowed'),
         'rlt_gpu4_gpu5_queued': value.get('rlt_queued'),
         'gpu_snapshot_csv': value['gpu'], 'gpu_context_snapshot': value['contexts'],
         'wm_batches': {'count': wm.get('count', 0), 'recent': recent}, 'worker_proofs': proofs,
         'reused_smoke': {'directory': 'runs/lift-two-gpu-b32-v2', 'exit_code': 0, 'seconds': 1766.130871752277,
                          'policy_batch': 64, 'wm_batch': 32, 'resource_only': True,
                          'short_C32_grad_norm': 0, 'not_evidence_of_learning': True},
         'prior_formal_failure': 'Optional nvidia-smi compute-memory query timeout during CP10 formal initialization; no formal update',
         'smoke_repeated': False}
dest = ROOT / 'docs/world-model/task_reward_plan_20261006/two_gpu_from0_light.json'
dest.write_text(json.dumps(light, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({'file': str(dest), 'observed_at': value['time']}))
