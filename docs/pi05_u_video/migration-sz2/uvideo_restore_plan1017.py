import sys,json,os,time,subprocess,shutil,fcntl,copy
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');D=Path('/data/chenyiteng/deployment-20261017/pi05-u-video-sz2-v2');O=Path('/data/chenyiteng/results/pi05-u-video/20261017-sz2-v2');sys.path.insert(0,str(S/'ops'));from common import read,save,same,proc,gpus
p=read(S/'plan.json');state=read(S/'status.json');assert not same(read(S/'owner-identity.json')) and not(D/'plan-restored-complete.json').exists()
assert not any((O/'batches').iterdir()) and all(not gpus()[g]['processes']for g in [0,4,5])
lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
R=Path(read(O/'manifest.json')['source']);updated=copy.deepcopy(p);A=D/'plan-restore-failed';A.mkdir(exist_ok=True)
for row in read(D/'rlt-preempt-complete.json')['rows']:
 g=row['gpu'];old=row['request'];rt=Path(old['runtime']);run=Path(old['run']);newrun=run.with_name(run.name+'-after-uvideo1017');newrt=newrun/'runtime'
 assert (not newrt.exists() or not any(newrt.iterdir())) and row['release']['cleanup_error']is None
 newrt.mkdir(parents=True,exist_ok=True)
 cfg=(rt/'resolved.yaml').read_text();(newrt/'resolved.yaml').write_text(cfg.replace(old['namespace'],newrun.name))
 env=read(rt/'environment.json');env={k:v.replace(str(run),str(newrun))if isinstance(v,str)else v for k,v in env.items()};save(newrt/'environment.json',env)
 q=dict(old,run=str(newrun),runtime=str(newrt),namespace=newrun.name);dest=D/f'fallback-g{g}.json';save(dest,q);updated['slots'][g]['fallback']=str(dest)
 assert Path(q['checkpoint'],'actor/sac_components/rlt_trainer_state/complete.json').is_file()
for g in ['4','5']:
 path=D/f'lane-g{g}-plan.json';assert not path.exists();shutil.copyfile(D/'bootstrap-failed'/path.name,path)
 lane=read(path);assert len(lane['jobs'])==50 and all(Path(j['config']).is_file()for j in lane['jobs'])
 for suffix in ['.log','-identity.json','-status.json','-error.json','-released.json']:
  path=D/f'lane-g{g}{suffix}'
  if path.exists():assert path.stat().st_uid==os.getuid();path.rename(A/path.name)
 with(D/f'lane-g{g}.log').open('x')as log:c=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_u_video/lane.py'),g],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',U_VIDEO_ROOT=str(O),U_VIDEO_TRANSACTION=str(D)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if(D/f'lane-g{g}-status.json').exists():break
  assert c.poll()is None;time.sleep(.1)
 ident=read(D/f'lane-g{g}-identity.json');assert same(ident) and read(D/f'lane-g{g}-status.json')['phase']in ['WAIT_PREDECESSOR','RUNNING'];updated['slots'][g]['external_identity']=ident;state['slots'][g]={'phase':'WAIT_EXTERNAL'}
assert all(updated['slots'][g]==p['slots'][g]for g in ['6','7']);save(S/'plan.json',updated);save(S/'status.json',state)
lock.close()
with(D/'scheduler-plan-restored.log').open('x')as log:c=subprocess.Popen(['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],cwd=S,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
save(D/'plan-restored-complete.json',dict(time=time.time(),owner=proc(c.pid),lanes={g:updated['slots'][g]['external_identity']for g in ['4','5']},fallbacks={g:read(updated['slots'][g]['fallback'])for g in ['4','5']},order='U/video -> same RLT configs/checkpoints in fresh dispatch directories',protected67=True))
print(json.dumps(read(D/'plan-restored-complete.json')))
