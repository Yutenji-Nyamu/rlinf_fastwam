set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import csv, datetime, json, math, re, subprocess
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc')
run=base/'pi0-adjust-bottle-smoke32x1-b1024-u10-eval16x2-gpu6-20260905-v7'
assert (run/'exit_code.txt').read_text().strip()=='0'
log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace'))
errors=[line for line in log.splitlines() if re.search(r'Fatal Python|Traceback \(most recent|OutOfMemoryError|ErrorInitializationFailed|cannot create buffer|pthread_key_create|AssertionError|RuntimeError:',line)]
assert not errors, errors[:3]
checkpoints=[]
for step in [1,2]:
    ckpt=run/f'pi0-bc-u10-smoke-gpu6/checkpoints/global_step_{step}/actor'
    files={str(p.relative_to(ckpt)):p.stat().st_size for p in ckpt.rglob('*') if p.is_file()}
    for rel in ['local_shard_checkpoint/checkpoint_rank_0.pt','model_state_dict/full_weights.pt','online_bc/rank_0/success_replay.pt','online_bc/rank_0/learner.pt']:
        assert files.get(rel,0)>0, (step,rel,files)
    learner=torch.load(ckpt/'online_bc/rank_0/learner.pt',map_location='cpu',weights_only=True)
    assert learner['update_step']==step*10, learner
    checkpoints.append({'step':step,'updates':learner['update_step'],'files':files})
ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
scalars={tag:[{'step':x.step,'value':x.value} for x in ea.Scalars(tag)] for tag in ea.Tags()['scalars'] if any(k in tag for k in ['success','bc/','grad_norm','time/step'])}
for tag in ['eval/success_once','env/success_once','train/bc/actor_loss','train/actor/grad_norm']:
    assert tag in scalars and len(scalars[tag])==2, (tag,scalars.keys())
    assert all(math.isfinite(x['value']) for x in scalars[tag]), tag
rows=list(csv.DictReader((run/'resource.csv').open()))
valid=[r for r in rows if r['gpu6_used_mib'].isdigit()]
result={'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),
    'optimizer_updates':20,'complete_rounds':2,'eval_episodes_per_round':32,
    'scalars':scalars,'checkpoints':checkpoints,'errors':errors,
    'sampled_peak_gpu_mib':max(int(r['gpu6_used_mib']) for r in valid),
    'min_host_available_gib':min(int(r['host_mem_available_kib'])/1024**2 for r in valid),
    'max_memory_psi':max(float(r['mem_psi_some_avg10']) for r in valid),
    'max_env_open_fds':max(int(r['env_open_fds']) for r in rows if r['env_open_fds'].isdigit()),
    'max_env_rss_gib':max(int(r['env_rss_kib'])/1024**2 for r in rows if r['env_rss_kib'].isdigit()),
    'started_at':(run/'started_at.txt').read_text().strip(),
    'finished_at':(run/'finished_at.txt').read_text().strip()}
packet=base/'implementation-u10-20260905'
(packet/'smoke-verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
for args in [
    ['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
    ['df','-h','/data'],['free','-h'],['ps','-p','602620,321933,322685','-o','pid,etime,stat,comm'],
]:
    print(subprocess.run(args,capture_output=True,text=True,timeout=20).stdout)
PY
