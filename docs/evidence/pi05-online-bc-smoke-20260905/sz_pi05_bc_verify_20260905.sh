set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import csv,datetime,json,math,re,subprocess,zipfile
from pathlib import Path
import torch
from safetensors import safe_open
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from rlinf.data.online_bc import SuccessReplay
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1')
assert (run/'exit_code.txt').read_text().strip()=='0','Smoke has not completed successfully.'
log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace'))
errors=[line for line in log.splitlines() if re.search(r'Fatal Python|Traceback \(most recent|OutOfMemoryError|ErrorInitializationFailed|cannot create buffer|pthread_key_create|AssertionError|RuntimeError:',line)]
assert not errors,errors[:3]
checkpoints=[]
for step in (1,2):
    ckpt=run/f'pi05-pillbottle-bc-u10-eval8x4-smoke-gpu6/checkpoints/global_step_{step}/actor'
    files={str(p.relative_to(ckpt)):p.stat().st_size for p in ckpt.rglob('*') if p.is_file()}
    for rel in ('local_shard_checkpoint/checkpoint_rank_0.pt','model_state_dict/full_weights.pt','online_bc/rank_0/success_replay.pt','online_bc/rank_0/learner.pt'):
        assert files.get(rel,0)>0,(step,rel)
    for rel in ('local_shard_checkpoint/checkpoint_rank_0.pt','model_state_dict/full_weights.pt'):
        with zipfile.ZipFile(ckpt/rel) as z: assert any(n.endswith('/data.pkl') for n in z.namelist())
    learner=torch.load(ckpt/'online_bc/rank_0/learner.pt',map_location='cpu',weights_only=True)
    assert learner['update_step']==step*10
    replay=torch.load(ckpt/'online_bc/rank_0/success_replay.pt',map_location='cpu',weights_only=True)
    assert replay['records'] and all('action_weights' not in r for r in replay['records'])
    assert all(r['action'].numel()==50*14 and r['tokenized_prompt'].shape==(200,) for r in replay['records'])
    checkpoints.append({'step':step,'updates':learner['update_step'],'files':files,'success_episodes':replay['episodes'],'queries':len(replay['records'])})
# mmap only selected model tensors: confirm actual expert change and a frozen
# VLM tensor without loading a second GPU model or claiming worker restoration.
full=torch.load(ckpt/'model_state_dict/full_weights.pt',map_location='cpu',weights_only=True,mmap=True)
frozen=next(n for n in full if '.paligemma.' in n and n.endswith('q_proj.weight'))
changes={}
with safe_open('/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab/model.safetensors',framework='pt',device='cpu') as original:
    for name in ('action_out_proj.weight','time_mlp_in.weight',frozen):
        before=original.get_tensor(name).to(full[name].dtype)
        changes[name]={'max_abs_delta':float((full[name]-before).abs().max()),'changed_elements':int((full[name]!=before).sum()),'elements':full[name].numel(),'dtype':str(full[name].dtype)}
assert changes['action_out_proj.weight']['changed_elements']>0
assert changes[frozen]['changed_elements']==0
del full
pool=SuccessReplay(1234,str(run/'unused_verification_archive'))
pool.load_checkpoint(ckpt/'online_bc/rank_0')
expected=pool.sample(8)
restored=SuccessReplay(0,str(run/'unused_verification_archive'))
restored.load_checkpoint(ckpt/'online_bc/rank_0')
actual=restored.sample(8)
for key in expected['forward_inputs']:torch.testing.assert_close(actual['forward_inputs'][key],expected['forward_inputs'][key])
ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
scalars={tag:[{'step':x.step,'value':x.value} for x in ea.Scalars(tag)] for tag in ea.Tags()['scalars'] if any(k in tag for k in ('success','bc/','grad_norm','time/step','time/actor_training','time/generate_rollouts','time/eval'))}
for tag in ('eval/success_once','env/success_once','train/bc/actor_loss','train/actor/grad_norm'):
    assert tag in scalars and len(scalars[tag])==2
    assert all(math.isfinite(x['value']) for x in scalars[tag])
rows=list(csv.DictReader((run/'resource.csv').open()))
valid=[r for r in rows if r['gpu6_used_mib'].isdigit()]
result={'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),'optimizer_updates':20,'complete_rounds':2,'eval_episodes_per_round':32,'eval_concurrent':8,'eval_batches':4,'scalars':scalars,'checkpoints':checkpoints,'errors':errors,'model_tensor_changes':changes,'replay_rng_readback':True,'sampled_peak_gpu_mib':max(int(r['gpu6_used_mib']) for r in valid),'min_host_available_gib':min(int(r['host_mem_available_kib'])/1024**2 for r in valid),'max_memory_psi':max(float(r['mem_psi_some_avg10']) for r in valid),'max_env_open_fds':max(int(r['env_open_fds']) for r in rows if r['env_open_fds'].isdigit()),'max_env_rss_gib':max(int(r['env_rss_kib'])/1024**2 for r in rows if r['env_rss_kib'].isdigit()),'started_at':(run/'started_at.txt').read_text().strip(),'finished_at':(run/'finished_at.txt').read_text().strip(),'scope':'Actual pi05 two-round collection/update/sync/eval8x4/checkpoint; CPU learner/replay RNG and selected full-model tensors readback. No full worker/optimizer restoration; no 100-round stability claim.'}
result['gpu6_now']=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(result))
PY
