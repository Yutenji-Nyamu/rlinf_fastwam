"""Prepare the isolated FastWAM EXPO experiment from two known production checkouts."""
import hashlib,json,math,os,shutil,subprocess,sys
from pathlib import Path
import yaml
E=Path(os.environ['EXPO_DEPLOYMENT'])
F=Path(os.environ['FASTWAM_GRPO_DEPLOYMENT'])
DATA=Path(os.environ['EXPO_OUTPUT'])
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    assert os.getuid()==E.stat().st_uid and Path(__file__).resolve().parent==E
    p=read(F/'plan.json'); live=Path(p['training_source']); source=E/'source'
    source.mkdir(exist_ok=True)
    if not (source/'rlinf').exists():
        shutil.copytree(live/'rlinf',source/'rlinf',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    native_patch = str(E/'fastwam-expo-native.patch')
    applied = subprocess.run(['git','apply','--reverse','--check',native_patch],cwd=source,capture_output=True)
    if applied.returncode:
        subprocess.run(['git','apply',native_patch],cwd=source,check=True)
    legacy=Path(os.environ['HISTORICAL_EXPO_SOURCE'])
    for base in ['rlinf/algorithms/expo_ft','examples/embodiment']:
        (source/base).mkdir(parents=True,exist_ok=True)
    for name in ['rlinf/algorithms/expo_ft','rlinf/algorithms/norm_signal.py','rlinf/algorithms/online_bc_dvac_controls.py','rlinf/algorithms/online_bc_dvac_two_level.py','rlinf/algorithms/ugrow_signal.py','examples/embodiment/train_expo_ft.py','examples/embodiment/train_expo_formal.py']:
        origin=legacy/name
        if origin.is_dir():shutil.copytree(origin,source/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
        elif origin.is_file():shutil.copy2(origin,source/name)
    for path in (E/'overlay').rglob('*.py'):
        dest=source/path.relative_to(E/'overlay');dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    raw=yaml.safe_load((F/'config/formal.yaml').read_text())
    c=read(E/'historical-inputs.json');c['model']=raw['actor']['model']
    assert c['model']['action_horizon']==32 and c['model']['num_action_chunks']==24
    c['physical_gpus']=[4,7];c['physical_gpu']=4;c['base_parameters']='FastWAM action expert; native FM objective'
    c['source_head']='FastWAM model and environment from current turn_switch run; clean EXPO algorithm'
    c.pop('norm_sha256',None);c.pop('rlt_policy',None)
    c['formal'].update(fm_microbatch=1,candidate_microbatch=2,observation_microbatch=2,max_physical_actions=20000,scope='20000 online training actions, evaluation excluded',prior_budget_truncations=0)
    c['core'].update(chunk_length=24,target_entropy=(24*14)*(math.log(c['core']['edit_scale'])-.5))
    for key,env in [('train',c['env']),('eval',c['evaluation']['config'])]:
        env.update(max_episode_steps=192,max_steps_per_rollout_epoch=192)
        if key=='eval':env.update(total_num_envs=1,rollout_epoch=20)
        env['task_config'].update(step_lim=192,ray_tracing_denoiser='none',save_path=str(DATA/('env-'+key)))
        env['video_cfg'].update(save_video=False,video_base_dir=str(DATA/'video'/key))
        assert Path(env['seeds_path']).exists()
    assert Path(c['demo_path'],'prepared.json').is_file() and len(list(Path(c['demo_path']).glob('data/*/episode_*.parquet')))==50
    assert all(Path(c['model'][key]).is_file() for key in ('model_path','dataset_stats_path'))
    c['port_source_manifest']={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted([*source.glob('examples/embodiment/train_expo*.py'),*source.glob('rlinf/algorithms/expo_ft/*.py'),*source.glob('rlinf/models/embodiment/fastwam/*.py')])}
    c['port_source_manifest'].update({str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (E/'overlay').rglob('*.py') for path in [source/path.relative_to(E/'overlay')]})
    write(E/'inputs.json',c);DATA.mkdir(parents=True,exist_ok=True)
    env=read(F/'training-environment.json')
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(uuids[i].strip() for i in (4,7)),PYTHONPATH=str(source)+':'+env.get('PYTHONPATH',''),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    import socket
    profile=Path.home()/'.nv/nvidia-application-profiles-rc.d/fastwam-expo47-1128.json'
    profile.parent.mkdir(parents=True,exist_ok=True)
    import xml.etree.ElementTree as ET
    minors=[int(node.findtext('minor_number')) for node in ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'])).findall('gpu')]
    visibility={'rules':[{'pattern':{'feature':'commname','matches':'fastwam-expo47'},'profile':['EGLVisibleDGPUDevices',sum(1<<minors[i] for i in (4,7))]}]}
    if profile.exists():assert read(profile)==visibility
    else:write(profile,visibility)
    (E/'scope-receipts').mkdir(exist_ok=True)
    write(E/'scope.json',dict(uid=os.getuid(),hostname=socket.gethostname(),gpu_uuids=[uuids[i].strip() for i in (4,7)],profile_path=str(profile),profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),commname='fastwam-expo47',receipt_directory=str(E/'scope-receipts')))
    env.update(PYTHONPATH=str(E/'bootstrap')+':'+env['PYTHONPATH'],RLINF_EXPO_GPU_SCOPE_MANIFEST=str(E/'scope.json'),__GL_APPLICATION_PROFILE='1')
    write(E/'environment.json',env)
    write(E/'plan.json',dict(gpus=[4,7],method='expo',bench='rt2',task='turn_switch',python=p['python'],training_source=str(source),run=str(DATA/'formal'),max_physical_actions=20000))
    print('ready',DATA,shutil.disk_usage(DATA).free//2**30,'GiB',len(c['port_source_manifest']),'files',sorted(env.keys()))
if __name__=='__main__':main()
