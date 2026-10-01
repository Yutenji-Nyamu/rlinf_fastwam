"""Read-only reconciliation of one genuine GRPO update in the completed two-epoch smoke.

Keeps the original stricter two-effective-epoch failure receipt unchanged. The
user's requested parameter-update smoke is evidenced by the second epoch only.
No training, checkpoint rewrite, Ray connection, or budget extension occurs.
"""
import hashlib,json,math,os,sys,time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/data/chenyiteng/projects/wan-goal-sz3')
W=R/'runs/wan-goal-sz3-20261001-r4'
S=W/'pi05-smoke';C=W/'pi05-smoke-control'
sys.path.insert(0,str(R/'scripts/resource_switch'))
from common import account,exclusive,sha

account()
verification=C/'smoke-verification.json'
v=json.loads(verification.read_text());metrics=v['metrics'];context=metrics['context']
assert v['audited_upstream_commit']=='d34d4c320d08cb982de034aa9a011f08dc0fa217'
assert v['ok'] is False and not v['errors'] and metrics['ok'] is False
assert metrics['errors']==['Non-finite rollout/advantages_min at step 0','Non-finite rollout/advantages_max at step 0']
assert context['runner_max_epochs']==2 and context['adv_type']=='grpo'
assert context['model_type']=='openpi' and context['train_expert_only'] is True
assert context['entropy_bonus']==0 and not context['enable_sft_co_train'] and not context['resume_dir']
empty,valid=metrics['epochs']
assert empty['tensorboard_step']==0 and empty['valid_rl_signal'] is False
assert empty['grad_norm']==0 and empty['total_loss']==0 and empty['loss_mask_fraction']==0
assert valid['tensorboard_step']==1 and valid['valid_rl_signal'] is True
for name in ('grad_norm','total_loss','loss_mask_fraction','advantages_min','advantages_max'):
    assert math.isfinite(valid[name]),name
assert valid['grad_norm']>0 and 0<valid['loss_mask_fraction']<=1
assert max(abs(valid['advantages_min']),abs(valid['advantages_max']))>0
assert v['process_exit_confirmed'] is True and v['weight_change']['ok'] is True
assert [x['step'] for x in v['checkpoints']]==[1,2] and all(x['ok'] for x in v['checkpoints'])
exit_path=S/'wm-exit.json';ex=json.loads(exit_path.read_text())
assert ex['exit_code']==0 and ex['outcome']=='completed' and ex['error'] is None
release_path=C/'wm-release.json';release=json.loads(release_path.read_text())
assert release['wm_exit_code']==0 and release['outcome']=='completed' and release['error'] is None
assert release['physical_gpus']==[4,5,6,7]
assert all(release[k] is True for k in ('all_workers_stopped','processes_clear','gpus_released'))
cleanup_path=Path(release['cleanup_receipt'])
assert sha(cleanup_path)==release['cleanup_receipt_sha256']
cleanup=json.loads(cleanup_path.read_text())
assert cleanup['owner_token']==release['owner_token'] and cleanup['cycle_id']==release['cycle_id']
assert cleanup['processes_clear'] and cleanup['gpus_released'] and cleanup['ray']['stopped']
placement_path=S/'verified-placement.json';placement=json.loads(placement_path.read_text())

# Inspect every floating saved policy tensor, not just the bounded delta sample.
import torch
finite=[]
for cp in v['checkpoints']:
    path=Path(cp['full_weights']);assert path.resolve().is_relative_to(W.resolve())
    state=torch.load(path,weights_only=True,mmap=True,map_location='cpu')
    count=elements=0
    for name,tensor in state.items():
        if isinstance(tensor,torch.Tensor) and tensor.is_floating_point():
            assert bool(torch.isfinite(tensor).all()),(cp['step'],name)
            count+=1;elements+=tensor.numel()
    finite.append({'step':cp['step'],'floating_tensors':count,'elements':elements,'all_finite':True})
    del state
assert not torch.cuda.is_initialized()

evidence_files=[verification,exit_path,release_path,cleanup_path,placement_path,
    Path(metrics['config_path']),*map(Path,metrics['event_files'])]
for cp in v['checkpoints']:
    evidence_files.append(Path(cp['actor_dir'])/'dcp_checkpoint/.metadata')
source_files=[R/'RLinf-pi05/rlinf/models/embodiment/openpi/policies/libero_policy.py',
    R/'RLinf-pi05/rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py']
receipt={
    'time':time.time(),'status':'ONE_VALID_PI05_GRPO_UPDATE_VERIFIED',
    'source_commit':v['audited_upstream_commit'],'learning_verified':True,'resources_released':True,
    'completed_runner_epochs':2,'effective_update_count':1,'valid_epoch':valid,'filtered_epoch':empty,
    'checkpoint_floating_parameters':finite,'actual_weight_change':v['weight_change'],
    'process_exit_confirmed':True,'original_strict_verifier_ok':False,
    'original_strict_verifier_preserved':str(verification),
    'evidence_sha256':{str(p):sha(p) for p in evidence_files},
    'approved_model_source_sha256':{str(p):sha(p) for p in source_files},
    'no_additional_smoke_budget':True,'formal_runner_epochs':1000,'formal_initialization':'original_fixed_SFT',
    'scope':'One genuine GRPO parameter update, complete CP1/CP2 and normal exit; not two effective updates or real LIBERO success.'}
output=C/'one-update-reconciled.json';exclusive(output,receipt)
print(json.dumps({'receipt':str(output),'sha256':sha(output),'learning_verified':True,
    'effective_update_count':1,'all_saved_floating_parameters_finite':True,'cuda_initialized':False}))
