#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import collections,datetime,hashlib,json,os,pickle,re,subprocess
from pathlib import Path
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
assert os.getuid()==1003
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
roots={
 'fastwam':base/'fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3',
 'sidney':base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1',
 'pi0':base/'grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2'}
s={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'torch':torch.__version__,'checkpoints':{},'ray_error_files':[],'crash_files':[]}
for name,root in roots.items():
    if name!='sidney':
        files=list(root.glob('*/checkpoints/global_step_*/actor/dcp_checkpoint/.metadata'))
        f=max(files,key=lambda p:int(p.parents[2].name.rsplit('_',1)[1]))
        with f.open('rb') as h:meta=pickle.load(h)
        tensors=[]
        for key,v in meta.state_dict_metadata.items():
            if hasattr(v,'properties') and hasattr(v.properties,'dtype'):
                tensors.append({'key':key,'dtype':str(v.properties.dtype),'shape':list(v.size)})
        s['checkpoints'][name]={'path':str(f),'metadata_bytes':f.stat().st_size,'tensors':tensors}
    else:
        files=list(root.glob('*/checkpoints/global_step_*/actor/local_shard_checkpoint/checkpoint_rank_0.pt'))
        f=max(files,key=lambda p:int(p.parents[2].name.rsplit('_',1)[1]))
        with FakeTensorMode():data=torch.load(f,map_location='cpu',mmap=True,weights_only=False)
        tensors=[]
        def walk(x,key=''):
            if isinstance(x,torch.Tensor):tensors.append({'key':key,'dtype':str(x.dtype),'shape':list(x.shape)})
            elif isinstance(x,dict):
                for k,v in x.items():walk(v,key+'.'+str(k))
            elif isinstance(x,(tuple,list)):
                for i,v in enumerate(x):walk(v,key+'.'+str(i))
        walk(data)
        s['checkpoints'][name]={'path':str(f),'top_keys':list(data),'tensors':tensors}
rayroots=[]
for pid in ['321933','322685']:
    try:
        args=Path('/proc',pid,'cmdline').read_bytes().decode().split('\0')
        for arg in args:
            if arg.startswith(('--log_dir=','--log-dir=')):rayroots.append(Path(arg.split('=',1)[1]))
    except OSError:pass
s['ray_log_roots']=[str(p) for p in rayroots]
for root in set(rayroots):
    for f in root.glob('*1569541*'):
        if f.suffix not in ['.err','.out']:continue
        lines=f.read_text(errors='replace').splitlines();hits=[i for i,l in enumerate(lines) if 'Fatal Python error' in l]
        s['ray_error_files'].append({'path':str(f),'bytes':f.stat().st_size,'fatal_raw':lines[max(0,hits[0]-100):hits[0]+900] if hits else lines[-20:]})
cr=Path('/var/crash')
if cr.exists():
    for f in cr.iterdir():
        st=f.stat()
        if st.st_uid==os.getuid():s['crash_files'].append({'path':str(f),'bytes':st.st_size,'mtime':st.st_mtime})
build=Path('/home/chenyiteng/builds/fastwam-scene-fence-20260904')
s['build_sources']=[str(p) for p in build.glob('*/*') if p.is_file() and p.suffix in ['.cc','.cpp','.c','.h','.hpp']]
s['native_hashes']={}
for f in [Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/sapien/pysapien.cpython-311-x86_64-linux-gnu.so'),Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/sapien.libs/libsvulkan2.so'),build/'release-final/librlinf_scene_fence.so']:
    with f.open('rb') as h:s['native_hashes'][str(f)]=hashlib.file_digest(h,'sha256').hexdigest()
s['native_source']={}
for f in [Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-clean-oidn-off-20260904/description/utils/generate_episode_instructions.py'),Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo/tools/fastwam_scene_fence/scene_fence.cpp')]:
    if f.exists():s['native_source'][str(f)]=f.read_text(errors='replace')
pysapien=Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/sapien/pysapien.cpython-311-x86_64-linux-gnu.so')
out=subprocess.run(['strings','-a',str(pysapien)],capture_output=True,text=True,timeout=20).stdout
s['pybind_abi_strings']=sorted(set(l for l in out.splitlines() if l.startswith(('__pybind11_internals','__pybind11_module_local'))))
for kind,args in [('shim_exports',['nm','-D','--defined-only']),('shim_python_imports',['nm','-D','--undefined-only'])]:
    out=subprocess.run(args+[str(build/'release-final/librlinf_scene_fence.so')],capture_output=True,text=True,timeout=15).stdout
    s[kind]=[l for l in out.splitlines() if kind=='shim_exports' or re.search('PyGIL|PyThread|PyEval|pthread',l)]
print(json.dumps(s,ensure_ascii=False))
PY
