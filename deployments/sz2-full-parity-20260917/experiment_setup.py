"""Select deployed SZ1 source profiles or prepare (never start) a new run."""
import argparse,copy,hashlib,json,os,pathlib,re,shlex,socket,subprocess,sys

BASE=pathlib.Path(__file__).resolve().parent
ALIASES={'grpo':'pi05-grpo-exp-controls-20260917','grpo-clean':'pi05-grpo-clean-half-seed42-20260911','bc':'pi05-bc8u5-fixed-clean-20260910','bc-dvac':'pi05-bc8u5-fixed-dvac-new-20260910','rlt':'rlt-clean-half600-20260914','rlt-tau':'rlt-t2-resume-t15-half4-800-20260916','stage1':'rlt-pi0-robotwin-ar-7d07a421','fastwam-bc':'fastwam-bc-turn-switch-20260914','fastwam-rlt':'fastwam-rlt-20260913','sarm':'pi05-online-bc-rynnvalue-rabc','iql':'pi05-online-iql-rynnvalue','attena':'pi05-online-bc-attena-fk'}
def load(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def flatten(x,key=''):
 if isinstance(x,dict):
  return {k:v for n,c in x.items() for k,v in flatten(c,key+'.'+n if key else n).items()}
 if isinstance(x,list):
  return {k:v for n,c in enumerate(x) for k,v in flatten(c,key+'.'+str(n)).items()}
 return {key:x}
def replace(x,old,new):
 if isinstance(x,dict):return {k:replace(v,old,new) for k,v in x.items()}
 if isinstance(x,list):return [replace(v,old,new) for v in x]
 return x.replace(old,new) if isinstance(x,str) else x
def choose(reg,value):
 value=ALIASES.get(value,value)
 exact=[p for p in reg['profiles'] if p['id']==value]
 if exact:return exact[0]
 candidates=[p for p in reg['profiles'] if p['tree']==value and not p['resume_reference'] and 'smoke' not in p['quality']]
 if value==ALIASES['stage1']:candidates=[p for p in candidates if p['stage1_training'] and '20260824-v2' in p['source_config']]
 if not candidates:raise SystemExit('No default captured formal profile for '+value+'; use list --tree '+value+' and the source configs.')
 candidates.sort(key=lambda p:('completed' in p['quality'],p['id']))
 return candidates[-1]
def environment(reg,tree,profile=None):
 env=load(BASE/'host-environment.json') if (BASE/'host-environment.json').exists() else {}
 if profile:env.update(load(profile['environment']))
 root=tree['root'];assets='/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support'
 env.update(VIRTUAL_ENV=reg['environment'],REPO_PATH=root,EMBODIED_PATH=root+'/examples/embodiment',RLINF_CODE_WORKING_DIR=root,PI05_REPO=root,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
 for key in ('ASSETS_PATH','ROBOTWIN_PATH','ROBOTWIN_ASSETS_PATH'):env.setdefault(key,assets)
 pp=env.get('PYTHONPATH',root+':'+assets)
 if root not in pp:pp=root+':'+pp
 if 'fastwam' in tree['name']:
  official=reg['external']['fastwam']['root']
  if official+'/src' not in pp:pp+=':'+official+'/src'
  env.update(FASTWAM_CONFIG_DIR=official+'/configs',DIFFSYNTH_MODEL_BASE_PATH='/data/chenyiteng/models/fastwam/diffsynth',DIFFSYNTH_DOWNLOAD_SOURCE='modelscope')
 env['PYTHONPATH']=pp
 env.setdefault('CUDA_HOME','/home/chenyiteng/tools/cuda-12.9')
 env['PATH']=reg['environment']+'/bin:'+env['CUDA_HOME']+'/bin:'+os.environ.get('PATH','')
 env.setdefault('LD_LIBRARY_PATH',env['CUDA_HOME']+'/lib64'+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
 env.setdefault('VK_DRIVER_FILES','/usr/share/vulkan/icd.d/nvidia_icd.json');env.setdefault('VK_ICD_FILENAMES',env['VK_DRIVER_FILES'])
 env.setdefault('OPENPI_DATA_HOME','/home/chenyiteng/.cache/openpi');env.setdefault('MUJOCO_GL','egl');env.setdefault('PYOPENGL_PLATFORM','egl');env.setdefault('ROBOT_PLATFORM','ALOHA');env.setdefault('TMPDIR','/data/chenyiteng/tmp')
 env['RYNNVALUE_PYTHON']=reg['rynnvalue_environment']+'/bin/python'
 env['EXPERIMENT_SETUP']=str(BASE/'experiment_setup.py');env['RLINF_PARITY_REGISTRY']=str(BASE/'registry.json')
 for mask in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(mask,None)
 return env
def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
 lp=sub.add_parser('list');lp.add_argument('--tree');lp.add_argument('--json',action='store_true')
 ep=sub.add_parser('env');ep.add_argument('tree')
 pp=sub.add_parser('prepare');pp.add_argument('profile',help='profile ID or family alias');pp.add_argument('--gpus',required=True,help='physical IDs, e.g. 1,2; must match source budget');pp.add_argument('--run',required=True);pp.add_argument('--namespace',required=True);pp.add_argument('--ray-address',required=True);pp.add_argument('--stage1-checkpoint');pp.add_argument('--stage1-manifest');pp.add_argument('--resume-checkpoint');pp.add_argument('--scorer-endpoint');pp.add_argument('--dry-run',action='store_true')
 a=ap.parse_args();reg=load(BASE/'registry.json')
 if a.cmd=='list':
  name=ALIASES.get(a.tree,a.tree);profiles=[r for r in reg['profiles'] if not name or r['tree']==name];trees=[r for r in reg['trees'] if not name or r['name']==name]
  if a.json:print(json.dumps({'trees':trees,'profiles':profiles},indent=2));return
  print('Aliases: '+', '.join(ALIASES))
  for t in trees:print(t['name']+' | '+t['head'][:12]+' | '+','.join(t['sz1_evidence_status']))
  print('\nCaptured profile IDs:')
  for p in profiles:print(p['id']+' | '+p['tree']+' | '+p['quality']+(' | resume' if p['resume_reference'] else ''))
  return
 if a.cmd=='env':
  name=ALIASES.get(a.tree,a.tree);tree=next((r for r in reg['trees'] if r['name']==name),None)
  if tree is None:raise SystemExit('Unknown tree '+name)
  try:profile=choose(reg,a.tree)
  except SystemExit:profile=None
  env=environment(reg,tree,profile)
  print('\n'.join('export '+k+'='+shlex.quote(str(v)) for k,v in env.items()))
  print('unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES')
  return
 profile=choose(reg,a.profile);tree=next(r for r in reg['trees'] if r['name']==profile['tree']);original=load(profile['config']);cfg=copy.deepcopy(original)
 ids=[int(x) for x in a.gpus.split(',')];assert len(ids)==len(set(ids)) and all(0<=i<8 for i in ids)
 placements=cfg['cluster']['component_placement'];assert len(placements)==1,'Multi-component placement needs an explicit reviewed source recipe.'
 old_gpu=str(next(iter(placements.values())))
 def expand(s):
  values=[]
  for item in s.split(','):
   if '-' in item:
    l,r=map(int,item.split('-'));values.extend(range(l,r+1))
   else:values.append(int(item))
  return values
 assert len(ids)==len(expand(old_gpu)),f'Source uses {old_gpu}; preserve GPU count.'
 out=pathlib.Path(a.run);assert out.resolve().is_relative_to(pathlib.Path('/data/chenyiteng/results').resolve()) and out.resolve()!=pathlib.Path('/data/chenyiteng/results').resolve() and not out.exists(),'Choose a fresh own result directory.'
 assert re.fullmatch('RLinf_[A-Za-z0-9_]+',a.namespace),'Use an explicit unique RLinf_... namespace.'
 oldrun=cfg['runner']['logger']['log_path'];cfg=replace(cfg,oldrun,str(out));cfg['runner']['logger']['experiment_name']=out.name
 cfg['cluster']['component_placement']={next(iter(placements)):','.join(map(str,ids))}
 if profile['resume_reference']:
  assert a.resume_checkpoint,'This is a resume profile: supply its complete local --resume-checkpoint.'
  assert pathlib.Path(a.resume_checkpoint).exists();cfg['runner']['resume_dir']=a.resume_checkpoint
 if profile['requires_stage1_output']:
  assert a.stage1_checkpoint and a.stage1_manifest,'RLT Stage2 requires your future Stage1 checkpoint and manifest.'
  ckpt=pathlib.Path(a.stage1_checkpoint);manifest=pathlib.Path(a.stage1_manifest);assert ckpt.exists() and manifest.is_file()
  cfg['rollout']['rlt_feature_model']['model_path']=str(ckpt)
  contract=cfg.setdefault('algorithm',{}).setdefault('rlt_resume',{}).setdefault('contract',{})
  m=load(manifest);contract['stage1_manifest_path']=str(manifest);contract['stage1_manifest_sha256']=sha(manifest)
  contract['stage1_manifest_id']=m.get('manifest_id',m.get('id',manifest.stem))
 if a.scorer_endpoint:
  assert a.scorer_endpoint.startswith('http://127.0.0.1:')
  for name in ('rabc','iql'):
   block=cfg.get('algorithm',{}).get('online_bc',{}).get(name)
   if block and 'endpoint' in block:block['endpoint']=a.scorer_endpoint
 env=environment(reg,tree,profile);env=replace(env,oldrun,str(out));env.update(RAY_ADDRESS=a.ray_address,CLUSTER_NAMESPACE=a.namespace)
 for key in ('NO_PROXY','no_proxy'):
  inherited=env.get(key,os.environ.get(key,''));parts=[p for p in inherited.split(',') if p]
  env[key]=','.join(dict.fromkeys(parts+['127.0.0.1','localhost',socket.gethostname()]))
 if a.stage1_manifest:env['RLT_STAGE1_MANIFEST_PATH']=a.stage1_manifest
 inputs=[]
 for key,val in flatten(cfg).items():
  if isinstance(val,str) and val.startswith('/') and any(s in key for s in ('model_path','checkpoint_path','dataset_path','dataset_stats_path','norm_stats_path','seeds_path','assets_path')):inputs.append({'key':key,'path':val,'exists':pathlib.Path(val).exists()})
 missing=[p for p in inputs if not p['exists']];assert not missing,json.dumps(missing,indent=2)
 entry=pathlib.Path(tree['root'])/profile['entry'];assert entry.is_file(),str(entry)
 diff={k:[flatten(original).get(k),v] for k,v in flatten(cfg).items() if flatten(original).get(k)!=v}
 runtime=out/'runtime';argv=[reg['environment']+'/bin/python','-u','-B',str(BASE/'prepared_driver.py'),str(runtime)]
 plan={'profile':profile,'repo':tree['root'],'head':tree['head'],'gpus':ids,'namespace':a.namespace,'ray_address':a.ray_address,'run':str(out),'entry':str(entry),'environment':env,'inputs':inputs,'diff':diff,'command':shlex.join(argv),'no_training_started':True,'stop_condition':'original configured endpoint; user stop or unrecoverable error'}
 if a.dry_run:print(json.dumps(plan,indent=2));return
 runtime.mkdir(parents=True)
 (runtime/'resolved.yaml').write_text(json.dumps(cfg,indent=2));(runtime/'plan.json').write_text(json.dumps(plan,indent=2));(runtime/'environment.json').write_text(json.dumps(env,indent=2));(runtime/'command.txt').write_text(shlex.join(argv)+'\n')
 print(json.dumps({'prepared':str(runtime),'command':shlex.join(argv),'training_started':False},indent=2))
if __name__=='__main__':main()
