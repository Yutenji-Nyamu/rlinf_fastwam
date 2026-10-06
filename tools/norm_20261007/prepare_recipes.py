"""Exact SZ1 algorithm configs plus Norm identity and SZ3 deployment routes."""
import ast,copy,hashlib,importlib.util,json,os,socket,subprocess,sys,tarfile
from pathlib import Path
import yaml
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
os.umask(0o077)
S=Path('/data/chenyiteng/projects/norm-bc-dsrl-sz3-20261007');CONTROL=S/'control';CONTROL.mkdir(exist_ok=True);assert not (CONTROL/'plan.json').exists()
ASSETS=Path('/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support')
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while data:=f.read(8*1024*1024):h.update(data)
    return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2))
def cmd(args):return subprocess.check_output(args,text=True,stderr=subprocess.PIPE,timeout=120).strip()
def mod(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
reference=read(S/'asset-identity.json')
assert cmd(['git','-C',str(ASSETS),'rev-parse','HEAD'])==reference['dsrl_env']['head']
for rel,digest in reference['dsrl_env']['python'].items():assert sha(ASSETS/rel)==digest,rel
model=Path('/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab')
model_digests={}
for rel,row in reference['model'].items():
    if rel=='conversion_manifest.json':
        assert read(model/rel)==read(S/'sz1-conversion-manifest.json'), 'Conversion metadata meaning differs'
    else:assert (model/rel).stat().st_size==row['bytes'] and sha(model/rel)==row['sha256'],rel
    model_digests[rel]={'content_equal':True,'comparison':'parsed JSON' if rel=='conversion_manifest.json' else 'SHA256'}
fixed=S/'robotwin-bc'
if not fixed.exists():cmd(['git','clone','--shared',str(ASSETS),str(fixed)])
with tarfile.open(S/'bc-env-python.tar.gz') as f:
    for m in f.getmembers():assert not m.issym() and not m.islnk() and (fixed/m.name).resolve().is_relative_to(fixed.resolve())
    f.extractall(fixed)
for rel,digest in reference['bc_env']['python'].items():assert sha(fixed/rel)==digest,rel
save(CONTROL/'assets-verified.json',{'model':reference['model'],'model_comparison':model_digests,'bc_env_head':reference['bc_env']['head'],'bc_env_python_count':len(reference['bc_env']['python']),'dsrl_env_head':reference['dsrl_env']['head'],'dsrl_env_python_count':len(reference['dsrl_env']['python'])})

# Reuse SZ1's proven one-card graphics/CUDA bootstrap with task-specific identity.
code=S/'dsrl/tools/dsrl_u_20261006/ops';sys.path.insert(0,str(code))
source=(code/'prepare_lease.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='scope')
text=ast.get_source_segment(source,node)
text=text.replace('libdsrl_u_g','libnorm_g').replace('_20261006.so','_20261007.so').replace('00-dsrl-u-g','00-norm-g').replace('-20261006.json','-20261007.json').replace("'uid': 1003, 'hostname': 'admin'","'uid': 20001, 'hostname': 'h100-gpu01'")
namespace={'Path':Path,'subprocess':subprocess,'sys':sys,'os':os,'sha':sha,'__file__':str(code/'prepare_lease.py'),'MASKS':('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES')}
def scope_save(p,v,exclusive=False):
    p=Path(p)
    if exclusive:assert not p.exists()
    p.parent.mkdir(parents=True,exist_ok=True);save(p,v)
namespace['save']=scope_save;exec(text,namespace)
manifest=read(S/'reference/manifest.json');cycle=S/'rlt-after-norm-g67-v1';prior=read(cycle/'plan.json')
base_env=read(cycle/'prepared/gpu6/environment.json')
norm=mod('norm_config_dsrl',S/'dsrl/rlinf/algorithms/dsrl_ugrow.py')
plan={'uid':20001,'hostname':'h100-gpu01','boot_id':prior['boot_id'],'ray_address':prior['ray_address'],'ray_dashboard_url':prior['ray_dashboard_url'],'python':PY,'root':str(S),'control':str(CONTROL),'cycle':str(cycle),'roles':{},'pins':{}}

def route(v,pairs):
    if isinstance(v,dict):return {k:route(x,pairs) for k,x in v.items()}
    if isinstance(v,list):return [route(x,pairs) for x in v]
    if isinstance(v,str):
        for a,b in pairs:v=v.replace(a,b)
    return v
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):return [x for k in sorted(set(a)|set(b)) for x in diff(a.get(k),b.get(k),path+'.'+k if path else k)]
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):return [x for i,(v,w) in enumerate(zip(a,b)) for x in diff(v,w,path+'.'+str(i))]
    return [] if a==b else [{'key':path,'old':a,'new':b}]

for role,gpu,total in [('bc',6,300),('dsrl',7,200)]:
    repo=S/role;old=yaml.safe_load((S/'reference'/role/'resolved.yaml').read_text())
    uuid=cmd(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'])
    scope_dir=CONTROL/('scope-g'+str(gpu))
    if (scope_dir/'scope.json').exists():
        scope=read(scope_dir/'scope.json');assert scope['uid']==20001 and scope['gpu_uuid']==uuid and scope['token']==CONTROL.name+'-g'+str(gpu)
        pins={str(scope_dir/'scope.json'):sha(scope_dir/'scope.json')}
        for key in ('marker','profile','bootstrap','runtime'):
            assert sha(scope[key+'_path'])==scope[key+'_sha256'];pins[scope[key+'_path']]=scope[key+'_sha256']
        env={k:v for k,v in base_env.items() if not k.startswith('RLT_')}
        env.update(REPO_PATH=str(repo),EMBODIED_PATH=str(repo/'examples/embodiment'),RLINF_CODE_WORKING_DIR=str(repo),RLINF_OPENDW_GPU_SCOPE_MANIFEST=str(scope_dir/'scope.json'),LD_PRELOAD=scope['marker_path'],__GL_APPLICATION_PROFILE='1')
    else:env,pins=namespace['scope'](CONTROL,gpu,uuid,base_env,repo,prior['repo'])
    env['ROBOTWIN_PATH']=str(fixed if role=='bc' else ASSETS)
    env['ASSETS_PATH']=str(ASSETS);env['ROBOTWIN_ASSETS_PATH']=str(ASSETS)
    env['PYTHONPATH']=':'.join([str(CONTROL/('scope-g'+str(gpu))/'bootstrap'),str(repo),env['ROBOTWIN_PATH']])
    env['PI05_MODEL_PATH']=str(model);env['RAY_ADDRESS']=prior['ray_address']
    plan['pins'].update(pins)
    item={'gpu':gpu,'gpu_uuid':uuid,'repo':str(repo),'baseline':manifest[role],'runs':{}}
    for phase,steps in [('smoke',2 if role=='bc' else 14),('formal',total)]:
        run=Path('/data/chenyiteng/results/norm-bc-dsrl-20261007')/(role+'-'+phase+str(total)+'-v1')
        rt=run/'runtime';rt.mkdir(parents=True,exist_ok=True);assert not (rt/'launch.json').exists()
        cfg=route(old,[(manifest[role]['repo'],str(repo)),(manifest[role]['run'],str(run)),(Path(manifest[role]['run']).name,run.name)])
        cfg['runner'].update(max_epochs=steps,max_steps=steps,resume_dir=None)
        if phase=='smoke':cfg['runner'].update(val_check_interval=0,save_interval=steps)
        if role=='bc':cfg['algorithm']['online_bc']['dvac']['signal_kind']='norm_residual_t5_l3'
        else:
            spec=norm.make_signal_spec(signal_kind='norm_residual_t5_l3',chunk_length=20)
            cfg['actor']['model']['openpi']['dsrl_u_spec']=spec
            cfg['algorithm']['dsrl_u']['spec']=copy.deepcopy(spec)
        name='norm_'+role+'_g'+str(gpu)
        cfg['cluster']['component_placement']={'actor,env,rollout':{'node_group':name,'placement':str(gpu)}}
        runtime_env=dict(env)
        if role=='bc':runtime_env['ONLINE_BC_RUN_DIR']=str(run)
        cfg['cluster']['node_groups']=[{'label':name,'node_ranks':'0','env_configs':[{'node_ranks':'0','python_interpreter_path':PY,'env_vars':[{k:v} for k,v in sorted(runtime_env.items())]}]}]
        for mode in ('train','eval'):
            original=Path(old['env'][mode]['seeds_path']).relative_to(manifest[role]['repo'])
            with tarfile.open(S/'reference'/role/'source.tar') as archive:
                frozen_seed=archive.extractfile(original.as_posix()).read()
            assert sha(repo/original)==hashlib.sha256(frozen_seed).hexdigest()
            assert Path(cfg['env'][mode]['seeds_path']).is_file()
        differences=diff(old,cfg)
        allowed=lambda k:k.startswith(('cluster.','actor.model.openpi.dsrl_u_spec.','algorithm.dsrl_u.spec.')) or k in {'env.train.seeds_path','env.eval.seeds_path','env.train.task_config.save_path','env.eval.task_config.save_path','env.train.video_cfg.video_base_dir','env.eval.video_cfg.video_base_dir','runner.logger.log_path','runner.logger.experiment_name','runner.max_epochs','runner.max_steps','algorithm.online_bc.data_path','algorithm.online_bc.dvac.signal_kind'} or (phase=='smoke' and k in {'runner.val_check_interval','runner.save_interval'})
        assert all(allowed(d['key']) for d in differences),[d for d in differences if not allowed(d['key'])]
        (rt/'resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));save(rt/'environment.json',runtime_env)
        save(CONTROL/(role+'-'+phase+'-config-diff.json'),differences)
        for p in [rt/'resolved.yaml',rt/'environment.json']:plan['pins'][str(p)]=sha(p)
        item['runs'][phase]={'run':str(run),'runtime':str(rt),'namespace':'norm-sz3-'+role+'-'+phase+'-1007-v1','steps':steps,'entry':'examples/embodiment/train_embodied_agent.py'}
    plan['roles'][role]=item
save(CONTROL/'plan.json',plan)
print(json.dumps({'control':str(CONTROL),'roles':plan['roles'],'assets_verified':True,'smoke_overrides':'serial rounds and eval/save timing only; warmup, UTD, N and batches unchanged'}))
