import json,math,os,sys,time
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=S/'trick-ablation-20261008';sys.path.insert(0,str(S/'ops'))
from common import read,same,gpus,proc,save
p=read(S/'plan.json');st=read(S/'status.json');pre=read(T/'prestop.json');out=dict(time=time.time(),host=p['host'],owner_alive=same(read(S/'owner-identity.json')),protected_unchanged=all(same(q) for q in pre['protected']),rows=[]);snap=gpus()
for g,slot in p['slots'].items():
 if 'priority' not in slot:continue
 q=read(slot['priority']);root=Path(q['run']);rt=root/'runtime';m={}
 for d,dirs,fs in os.walk(root):
  dirs[:]=[n for n in dirs if n not in ['checkpoints','success_data','video','robotwin_data','replay_buffer']]
  for n in fs:
   if not n.startswith('events.out.tfevents'):continue
   a=EventAccumulator(str(Path(d)/n),size_guidance={'scalars':10});a.Reload()
   for tag in a.Tags()['scalars']:
    if any(x in tag for x in ['success_once','bc_signal_tau','bc/actor_loss','actor/grad_norm','bc/grad','replay_buffer/size','bc/update','actor/loss','train/online_bc']):
     v=a.Scalars(tag)[-1];m[tag]=dict(step=v.step,value=v.value)
 row=st['slots'][g];f=rt/'driver.log';tail=''
 if f.exists():
  with f.open('rb') as z:z.seek(max(0,f.stat().st_size-3000));tail=z.read().decode(errors='replace')
 bindings=[]
 for idx,card in snap.items():
  for process in card['processes']:
   a=proc(process['pid'])
   if not a or a['uid']!=os.getuid():continue
   try:ev=dict(x.split('=',1) for x in (Path('/proc')/str(a['pid'])/'environ').read_bytes().decode(errors='replace').split('\0') if '='in x)
   except OSError:continue
   if ev.get('CLUSTER_NAMESPACE')==q['namespace']:bindings.append(dict(gpu=idx,**process))
 out['rows'].append(dict(gpu=g,namespace=q['namespace'],phase=row['phase'],driver_alive=same(row['identity']) if 'identity'in row else False,identity=row.get('identity'),metrics=m,bindings=bindings,finished=read(rt/'finished.json') if (rt/'finished.json').exists() else None,tail=tail))
out['gpu_0_3']= {g:snap[g] for g in range(4)};save(T/'latest-probe.json',out)
print(json.dumps(out))
