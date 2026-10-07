import collections,json,os,socket,sys,time,hashlib,subprocess
from pathlib import Path
R=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001');T=R/'formal-turn-switch-repair-20261002';C=R/'parallel-trial-20261006'
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu02' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
out={'time':time.time(),'events':[]}
for line in (T/'run/events.jsonl').read_text().splitlines():
 try:e=json.loads(line)
 except json.JSONDecodeError:continue
 if e['event'] in ('real_chunk','base_fm_source','checkpoint_committed','evaluation_native_enter','evaluation_finished','episode_started','episode_finished','learner_finished'):
  if e['event']=='checkpoint_committed':e={k:e[k] for k in ('event','time','reason','core_updates','cadence')}
  elif e['event']=='episode_finished':e={k:v for k,v in e.items() if k!='replay_entry'}
  out['events'].append(e)
idx=json.loads((T/'replay/index.json').read_text());bad=[]
for e in idx['online_entries']:
 for k,pk in [('path','pin'),('manifest_path','manifest_pin')]:
  p=T/'replay'/e[k];s=p.stat();actual=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid]
  if actual!=e[pk]['identity']:bad.append(e[k])
out['replay_file_identity']={'checked_files':2*len(idx['online_entries']),'mismatches':bad,'payload_hash_reread':False}
inp=json.loads((T/'inputs.json').read_text());demo=Path(inp['demo_path']);out['demo_prepared']=json.loads((demo/'prepared.json').read_text())
out['old_checkpoints']={}
for folder in [R/'continue-60k-20261005/baseline',C/'baseline']:
 out['old_checkpoints'][str(folder)]=[{'name':p.name,'bytes':p.stat().st_size} for p in folder.glob('*.pt')]
src=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-q-signals-20261006/tools/rlt_q_signals/coexist_owner.py')
out['coexist_source']=src.read_text() if src.exists() else None
out['latest_status']=json.loads((T/'run/status.json').read_text())
print(json.dumps(out))
