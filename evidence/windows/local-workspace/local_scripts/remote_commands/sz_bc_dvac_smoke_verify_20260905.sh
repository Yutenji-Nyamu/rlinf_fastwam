set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc-dvac
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root" CUDA_VISIBLE_DEVICES=""
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
"""Read-only verification of this smoke's own artifacts; never resumes a job."""
import csv, datetime, json, re, subprocess
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from rlinf.algorithms.online_bc_dvac import OnlineBCDvac
from rlinf.data.online_bc import SuccessReplay, masked_fm_loss

run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1')
assert (run/'exit_code.txt').read_text().strip()=='0', 'Smoke has not exited successfully; no passing verdict.'
log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace'))
errors=[s for s in ('CUDA out of memory','OutOfMemoryError','Fatal Python error','OIDN Error','pthread_key_create failed','RuntimeError:','ValueError:','Traceback','Exiting main process due to a failure') if s in log]
assert not errors, errors
steps=re.findall(r'Global Step:\s*(\d+)\s*/',log)
assert int(steps[-1])==2, steps[-5:]
ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
scalars={k:[[e.step,e.value] for e in ea.Scalars(k)] for k in ea.Tags().get('scalars',[]) if any(x in k for x in ('success','dvac','loss','grad_norm','time/step'))}
assert len(scalars['env/success_once'])==2 and len(scalars['eval/success_once'])==2
checkpoints=[]
previous_records=None
for n in (1,2):
    matches=list(run.glob(f'checkpoints/global_step_{n}/actor'))
    if not matches:
        matches=[p.parent.parent.parent for p in run.rglob('dvac.pt') if f'global_step_{n}' in p.parts]
    assert len(matches)==1, (n,matches)
    actor=matches[0]
    target=actor/'online_bc/rank_0'
    paths=('model_state_dict/full_weights.pt','local_shard_checkpoint/checkpoint_rank_0.pt','online_bc/rank_0/success_replay.pt','online_bc/rank_0/learner.pt','online_bc/rank_0/dvac.pt')
    files={rel:(actor/rel).stat().st_size for rel in paths}
    assert all(size>0 for size in files.values()), files
    learner=torch.load(target/'learner.pt',map_location='cpu',weights_only=True)
    assert learner['update_step']==n*10, learner
    state=torch.load(target/'dvac.pt',map_location='cpu',weights_only=True)
    calibrator=OnlineBCDvac(**state['settings']);calibrator.load_state_dict(state)
    assert calibrator.round_id==n and len(calibrator.history)==n
    replay=SuccessReplay(seed=0,archive_path=str(run/'unused-verification-target'))
    replay.load_checkpoint(target)
    records=replay.records
    assert records and all(r['dvac_v'].shape==(50,) and r['action_weights'].shape==(50,) and r['action_valid_mask'].shape==(50,14) for r in records)
    weight=torch.stack([r['action_weights'] for r in records])
    mask=torch.stack([r['action_valid_mask'] for r in records])
    q=mask.sum(-1)
    assert torch.isfinite(weight).all() and (weight>=0).all() and (weight<=2).all()
    assert torch.allclose((weight*q).sum(-1)/q.sum(-1),torch.ones(len(records)),atol=1e-6)
    if n==1:
        assert torch.equal(weight,torch.ones_like(weight))
    else:
        assert previous_records is not None
        assert all(torch.equal(a['action_weights'],b['action_weights']) and torch.equal(a['dvac_v'],b['dvac_v']) for a,b in zip(previous_records,records[:len(previous_records)])), 'Old replay weights changed.'
    fresh=[r for r in records if int(r['dvac_calibration_round'])==n-1]
    assert fresh
    before=OnlineBCDvac(**state['settings'])
    before.load_state_dict({'settings':state['settings'],'round_id':n-1,'history':state['history'][:-1]})
    annotations=[dict(r) for r in fresh]
    before.annotate([annotations],state['history'][-1])
    assert all(torch.equal(a['action_weights'],b['action_weights']) for a,b in zip(annotations,fresh)), 'Stored weights disagree with previous-round calibration.'
    if n==2:
        assert any(not torch.equal(r['action_weights'],torch.ones(50)) for r in fresh), 'Second round never applied nonuniform weights.'
    # Test sidecar replay RNG restored, without touching the saved file or GPU.
    again=SuccessReplay(seed=123,archive_path=str(run/'unused-verification-target'))
    again.load_checkpoint(target)
    a,b=replay.sample(32)['forward_inputs'],again.sample(32)['forward_inputs']
    assert all(torch.equal(a[k],b[k]) for k in a)
    loss=torch.arange(1,32*50*14+1,dtype=torch.float32).reshape(32,50,14)/(32*50*14)
    assert torch.isfinite(masked_fm_loss(loss,a['action_valid_mask'],a['action_weights']))
    fresh_weight=torch.stack([r['action_weights'] for r in fresh])
    checkpoints.append({'step':n,'updates':learner['update_step'],'actor_path':str(actor),'files':files,'episodes':replay.episodes,'queries':len(records),'fresh_queries':len(fresh),'calibration_round':calibrator.round_id,'stats_positions':[x[0].item() for x in calibrator.history],'weight_min':weight.min().item(),'weight_max':weight.max().item(),'weight_mean':weight.mean().item(),'weight_std':weight.std(unbiased=False).item(),'fresh_weight_mean_h':fresh_weight.mean(0).tolist(),'sidecar_and_replay_cpu_readback':True})
    previous_records=records
rows=list(csv.DictReader((run/'resource.csv').open()))
def maximum(k):return max([float(r[k]) for r in rows if r.get(k)]+[0])
result={'passed':True,'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'complete_rounds':2,'optimizer_updates':20,'train_attempts':64,'eval_attempts':64,'scalars':scalars,'checkpoints':checkpoints,'errors':errors,'sampled_gpu_peak_mib':maximum('gpu7_used_mib'),'ram_available_min_gib':min(float(r['host_mem_available_kib']) for r in rows if r.get('host_mem_available_kib'))/1024**2,'env_fd_max':maximum('env_open_fds'),'env_rss_peak_gib':maximum('env_rss_kib')/1024**2,'started_at':(run/'started_at.txt').read_text().strip(),'finished_at':(run/'finished_at.txt').read_text().strip(),'verification_scope':'Completed real collection/update/sync/eval/save. CPU sidecar/replay restore and recalibration checked. No full production worker/model/optimizer restart and no long-run OOM fix claimed.'}
result['gpu7_now']=subprocess.run(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
result['source_head']=subprocess.run(['git','-C','/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc-dvac','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
print(json.dumps(result,ensure_ascii=False))
PY
