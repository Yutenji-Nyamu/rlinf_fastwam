set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc-dvac
export PYTHONPATH="$root" REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment"
export ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export PI05_MODEL_PATH=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/server-maintenance-20260906/cpu-only-config-placeholder
cd "$root"
nice -n 19 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,json,os
from pathlib import Path
import torch
from rlinf.algorithms.online_bc_dvac import OnlineBCDvac,log_moments
from rlinf.data.online_bc import masked_fm_loss
torch.set_num_threads(2)
base=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc')
runs={'bc':base/'pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1',
'dvac':base/'pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1'}
result={'time':datetime.datetime.now().astimezone().isoformat(),'cuda_available':torch.cuda.is_available(),'runs':{}}
for name,run in runs.items():
 files=sorted((run/'success_data/rank_0').glob('batch_*.pt'))
 infos=[]
 for path in [files[0],files[1],files[-1]]:
  eps=torch.load(path,map_location='cpu',weights_only=True)
  rows=[r for e in eps for r in e]
  item={'path':str(path),'episodes':len(eps),'queries':len(rows),'keys':list(rows[0]),
    'action_shape':list(rows[0]['action'].shape),'mask_shape':list(rows[0]['action_valid_mask'].shape),
    'versions':sorted({float(v) for r in rows if r.get('policy_version') is not None for v in torch.as_tensor(r['policy_version']).flatten()}),
    'query_order_ok':all([int(r['query_idx']) for r in e]==list(range(len(e))) for e in eps),
    'all_finite_actions':all(torch.isfinite(r['action']).all().item() for r in rows),
    'mask_valid':all(r['action_valid_mask'].all().item() for r in rows)}
  if 'action_weights' in rows[0]:
   w=torch.stack([r['action_weights'] for r in rows]);v=torch.stack([r['dvac_v'] for r in rows])
   assert w.shape==v.shape and w.shape[1]==50
   item.update(weight_min=float(w.min()),weight_max=float(w.max()),weight_std=float(w.std(unbiased=False)),
    max_mean_error=float((w.mean(1)-1).abs().max()),all_unit=bool(w.eq(1).all()),
    calibration_rounds=sorted({int(r['dvac_calibration_round']) for r in rows}),
    weight_horizon_mean=w.mean(0).tolist(),variance_horizon_mean=v.mean(0).tolist(),
    all_variance_nonnegative=bool(v.ge(0).all()),correlation_logv_weight=float(torch.corrcoef(torch.stack([(v+1e-12).log().flatten(),w.flatten()]))[0,1]) if w.std()>0 else None)
  infos.append(item)
 result['runs'][name]=infos
 # Read only tiny learner sidecars, never the model or full replay checkpoint.
 result[name+'_learner_states']=[{'path':str(p),'state':torch.load(p,map_location='cpu',weights_only=True)} for p in run.glob('*/checkpoints/global_step_*/actor/online_bc/rank_0/learner.pt')]
print('CPU_RECORD_AUDIT_JSON='+json.dumps(result,ensure_ascii=False))
PY
