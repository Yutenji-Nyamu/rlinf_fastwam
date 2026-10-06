import importlib.util,json,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261006/ugrow-bc8u5-300-v1');D=S/'bc'
def read(p):return json.loads(Path(p).read_text()) if Path(p).is_file() else None
p=read(D/'plan.json');sp=importlib.util.spec_from_file_location('rt',p['ops']);rt=importlib.util.module_from_spec(sp);sp.loader.exec_module(rt);op,b=rt.configure(D)
state=read(D/'status.json');out={'time':time.time(),'cutover':read(S/'cutover-status.json'),'owner':state,'owner_alive':bool(state and b.same(state['owner'])),'log_tail':(D/'owner.log').read_text()[-3000:] if (D/'owner.log').is_file() else None,'runs':{},'gpu':b.gpu_snapshot()}
for k,row in p['runs'].items():
 r=Path(row['run']);m=rt.scalars(r);log=r/'runtime/driver.log'
 out['runs'][k]={'metrics':{tag:v[-1] for tag,v in m.items() if v and (tag in ('env/success_once','train/bc/actor_loss','train/actor/grad_norm','train/ugrow/weight_std','train/replay_buffer/filtered_success_episodes','train/replay_buffer/success_episodes','train/replay_buffer/query_records','train/ugrow/round'))},'finished':read(r/'runtime/finished.json'),'log_age':time.time()-log.stat().st_mtime if log.exists() else None}
 if log.exists():
  with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2500));out['runs'][k]['tail']=f.read().decode(errors='replace')
print(json.dumps(out))
