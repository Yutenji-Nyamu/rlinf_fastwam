import os,sys,time,json,signal,subprocess,copy,fcntl,shutil
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');D=Path('/data/chenyiteng/deployment-20261017/pi05-u-video-sz2-v1');O=Path('/data/chenyiteng/results/pi05-u-video/20261017-sz2-v1');N=D.with_name('pi05-u-video-sz2-v2');Q=O.with_name('20261017-sz2-v2');scope=Path('/data/chenyiteng/deployment-20261014/grpo-u-pillbottle-tau123-v1/scope-tau2')
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal,gpus
assert os.getuid()==20001 and not N.exists() and not Q.exists()
plan=read(S/'plan.json');owner=read(S/'owner-identity.json');snap=gpus();protected=[plan['slots'][g]['external_identity']for g in ['6','7']]
assert same(owner) and all(same(i)for i in protected)
children=[read(D/f'lane-g{g}-status.json')['child']for g in [4,5]]
assert all(same(i)for i in children)
assert all(row['pid']in[i['pid']for i in children] and row['type']=='G'for row in snap[0]['processes']) and snap[0]['processes']
N.mkdir();(Q/'configs').mkdir(parents=True);(Q/'batches').mkdir();(N/'scope-receipts').mkdir();(N/'bootstrap').mkdir()
manifest=read(scope/'scope.json');manifest['receipts_dir']=str(N/'scope-receipts');save(N/'graphics-scope.json',manifest)
# Same already-validated SZ2 scope implementation; CPU validation never enables it.
(N/'bootstrap/sitecustomize.py').write_text("import os\nif os.environ.get('RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST') and os.environ.get('CUDA_VISIBLE_DEVICES'):\n from graphics_scope_runtime import install\n install(os.environ['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'])\n")
for p in (O/'configs').glob('*.json'):
 if p.name.endswith('_seeds.json'):shutil.copyfile(p,Q/'configs'/p.name)
 else:
  c=read(p);c['output']=c['output'].replace(str(O),str(Q));c['env']['seeds_path']=c['env']['seeds_path'].replace(str(O),str(Q));c['env']['task_config']['save_path']=c['env']['task_config']['save_path'].replace(str(O),str(Q));c['env']['video_cfg']['video_base_dir']=c['env']['video_cfg']['video_base_dir'].replace(str(O),str(Q));save(Q/'configs'/p.name,c)
for g in [4,5]:
 e=read(O/f'environment-g{g}.json');e.update(PYTHONPATH=str(N/'bootstrap')+':'+str(scope/'bootstrap')+':'+e['REPO_PATH']+':'+e['ROBOTWIN_PATH'],TMPDIR=str(N/f'tmp-g{g}'),RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST=str(N/'graphics-scope.json'),LD_PRELOAD=str(scope/'libscope.so'),__GL_APPLICATION_PROFILE='1');Path(e['TMPDIR']).mkdir();save(Q/f'environment-g{g}.json',e)
 lane=read(D/f'lane-g{g}-plan.json');lane['jobs']=[dict(j,config=j['config'].replace(str(O),str(Q)))for j in lane['jobs']];save(N/f'lane-g{g}-plan.json',lane)
m=read(D/'prepared.json');m.update(time=time.time(),output=str(Q),plan_before=plan,corrects_startup=str(D),reason='Reuse existing SZ2 complete graphics scope after initial launch exposed graphics-only context on GPU0');save(N/'prepared.json',m);save(Q/'manifest.json',m)
args=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0');assert str(S/'ops/owner.py')in args
env=dict(x.split('=',1)for x in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0')if '='in x);cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
save(N/'cutover-started.json',dict(time=time.time(),owner=owner,plan=plan,graphics_before=snap,old_children=children))
exact_signal(owner,signal.SIGTERM)
for _ in range(50):
 if not same(owner):break
 time.sleep(1)
assert not same(owner);lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 for g in ['4','5']:exact_signal(plan['slots'][g]['external_identity'],signal.SIGTERM)
 for i in children:exact_signal(i,signal.SIGTERM)
 for _ in range(60):
  snap=gpus()
  if all(not same(i)for i in children) and all(not snap[g]['processes']for g in [0,4,5]):break
  time.sleep(1)
 else:raise RuntimeError('Initial scoped launch did not clear; no new launches')
 assert all(same(i)for i in protected)
 save(D/'superseded.json',dict(time=time.time(),replacement=str(N),graphics_cleared=True,initial_batch_outputs_retained=True))
 updated=copy.deepcopy(plan);status=read(S/'status.json');R=Path(m['source'])
 for g in ['4','5']:
  with(N/f'lane-g{g}.log').open('x')as log:c=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_u_video/lane.py'),g],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',U_VIDEO_ROOT=str(Q),U_VIDEO_TRANSACTION=str(N)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(100):
   if(N/f'lane-g{g}-identity.json').exists():break
   assert c.poll()is None;time.sleep(.1)
  ident=read(N/f'lane-g{g}-identity.json');assert same(ident);updated['slots'][g]['external_identity']=ident;updated['slots'][g]['external_release']=str(N/f'lane-g{g}-released.json');status['slots'][g]={'phase':'WAIT_EXTERNAL'}
  save(S/'plan.json',updated);save(S/'status.json',status)
 save(N/'cutover-complete.json',dict(time=time.time(),gpus=[4,5],fallbacks_preserved=all(updated['slots'][g]['fallback']==plan['slots'][g]['fallback']for g in plan['slots']),protected_67_unchanged=all(updated['slots'][g]==plan['slots'][g]for g in ['6','7'])))
finally:
 lock.close()
 with(N/'scheduler-after.log').open('x')as log:c=subprocess.Popen(args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(N/'scheduler-restarted.json',dict(time=time.time(),identity=proc(c.pid)))
print(json.dumps(read(N/'cutover-complete.json')))
