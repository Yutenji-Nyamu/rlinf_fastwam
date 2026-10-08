"""Add GPU4/5 to the existing per-card queue; no WMRL code changes."""
import os,json,sys,copy,time,signal,subprocess,importlib.util
from pathlib import Path
from omegaconf import OmegaConf
C=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');W=S/'wm-cycle-20261009-v2'
sys.path.insert(0,str(C/'ops'))
from common import read,save,same,proc,exact_signal
assert os.getuid()==20001
plan=read(C/'plan.json');owner=read(C/'owner-identity.json');wm=read(W/'run/owner.json');state=read(C/'status.json')
assert same(owner) and same(wm) and not (W/'run/final.json').exists()
assert set(plan['slots'])=={'6','7'}
protected=[wm]+[state['slots'][g]['identity'] for g in ['6','7']]
assert all(same(x) for x in protected)
old=read(S/'lift-two-gpu-b16-lean-20261007-v1/cycles/cycle/plan.json')
fragment=read(S/'rlt45-return-20261007-v1/scope/environment-fragment.json')
scope=read(fragment['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'])
requests={}
for g in ['4','5']:
 row=old['runs']['gpu'+g];prior=Path(row['new_run']);prior_rt=prior/'runtime'
 ns='rlt-sz3-g'+g+'-after-wm-cycle-1009';run=Path('/data/chenyiteng/results/rlinf-rlt')/ns;rt=run/'runtime'
 assert not (rt/'started.json').exists();rt.mkdir(parents=True,exist_ok=True)
 cfg=OmegaConf.to_container(OmegaConf.load(prior_rt/'resolved.yaml'),resolve=False)
 cp=row['recovery']['checkpoint'];assert (Path(cp['path'])/'actor/sac_components/rlt_trainer_state/complete.json').is_file()
 cfg=json.loads(json.dumps(cfg).replace(str(prior),str(run)))
 cfg['runner']['logger'].update(log_path=str(run),experiment_name=ns)
 assert cfg['runner']['resume_dir']==cp['path']
 assert cfg['cluster']['component_placement']=={'actor,env,rollout':int(g)} and cfg['env']['train']['total_num_envs']==8
 save(rt/'resolved.yaml',cfg)
 env=read(prior_rt/'environment.json');repo=env['REPO_PATH']
 env.update(fragment)
 env['PYTHONPATH']=fragment['PYTHONPATH'].split(':')[0]+':'+repo+':'+env['ROBOTWIN_PATH']
 env.pop('RLINF_OPENDW_GPU_SCOPE_MANIFEST',None)
 env['RLT_LOG_ROOT']=str(run);save(rt/'environment.json',env)
 request=C/'requests'/('rlt-g'+g+'-wmcycle1009.json')
 save(request,dict(kind='rlt',gpu=int(g),repo=repo,run=str(run),runtime=str(rt),namespace=ns,checkpoint=cp['path'],step=cp['step']))
 requests[g]=str(request)
 plan['slots'][g]=dict(gpu_uuid=scope['cards'][g]['gpu_uuid'],fallback=str(request),external_identity=wm,external_release=str(W/'run/final.json'))
# Reuse the established namespace registration, without altering queue/algorithm code.
spec=importlib.util.spec_from_file_location('old_owner',S/'lift-two-gpu-b16-lean-20261007-v1/code/owner.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
p=read(W/'plan.json');origin=read(p['origin_plan']);api.load_lifecycle(origin)
(W/'rlt-queue').mkdir(exist_ok=True)
api.add_allowlist(dict(origin,owner_dir=str(W/'rlt-queue'),management_namespace='wm-cycle-rlt-queue-1009',trials=[dict(namespace=read(q)['namespace']) for q in requests.values()]))
# Only reload the CPU queue owner. All training drivers keep their PID/start identity.
save(W/'rlt-queue/plan-before.json',read(C/'plan.json'))
exact_signal(owner,signal.SIGTERM)
deadline=time.monotonic()+55
while same(owner) and time.monotonic()<deadline:time.sleep(.5)
assert not same(owner),'CPU queue owner has not exited; training untouched'
save(C/'plan.json',plan)
enabled=read(C/'fallback-enabled.json');enabled['requests'].update(requests);enabled['time']=time.time();save(C/'fallback-enabled.json',enabled)
with (C/'owner.log').open('a') as log:
 child=subprocess.Popen([plan['python'],'-u','-B',str(C/'ops/owner.py'),'--plan',str(C/'plan.json')],cwd=str(C),env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
deadline=time.monotonic()+40
while time.monotonic()<deadline:
 current=read(C/'status.json')
 if current.get('owner',{}).get('pid')==child.pid and all(current['slots'].get(g,{}).get('phase')=='WAIT_EXTERNAL' for g in ['4','5']):break
 if child.poll() is not None:raise RuntimeError('Queue owner exited')
 time.sleep(1)
else:raise TimeoutError('Queue status not yet refreshed')
assert all(same(x) for x in protected)
receipt=dict(time=time.strftime('%F %T'),owner=proc(child.pid),wm_unchanged=wm,slots={g:current['slots'][g] for g in ['4','5']},bc67_unchanged=True,requests=requests,release=str(W/'run/final.json'))
save(W/'rlt-queue/added.json',receipt);print(json.dumps(receipt),flush=True)
