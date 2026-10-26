import json, os, sys, subprocess, time, xml.etree.ElementTree as ET
from pathlib import Path
D=Path('/data/chenyiteng/deployment-20261026/fql-smoke-sz1-g23');R=Path('/data/chenyiteng/deployment-20261026/fql-cpu/source')
ref=Path('/data/chenyiteng/deployment-20261025/endpoint-multitask-c10-sz1-g45')
assert os.getuid()==1003 and not D.exists()
x=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True));gpus=x.findall('gpu')
assert all(not gpus[g].findall('processes/process_info') and int(gpus[g].findtext('fb_memory_usage/used').split()[0])<100 for g in [2,3]), 'GPU2/3 no longer free'
D.mkdir();(D/'bootstrap').mkdir()
(D/'bootstrap/sitecustomize.py').write_text('import os,ctypes\ng=os.environ.get("DOJO_GPU_SCOPE")\nif g is not None:\n assert g in ("2","3")\n assert ctypes.CDLL(None).prctl(15,ctypes.c_char_p(("fql26-g"+g).encode()),0,0,0)==0\n')
profile=Path('/home/chenyiteng/.nv/nvidia-application-profiles-rc.d/00-fql-smoke-g23-20261026.json');assert not profile.exists()
profile.write_text(json.dumps({'rules':[{'pattern':{'feature':'commname','matches':f'fql26-g{g}'},'profile':['EGLVisibleDGPUDevices',1<<g]}for g in [2,3]]}))
o=Path(json.loads((ref/'prepared.json').read_text())['output']);base=json.loads((o/'environment-g4.json').read_text())
sys.path.insert(0,str(R/'tools/pi05_distillation'));from fql_prepare import prepare
configs=prepare(ref,D/'collection',2,['adjust_bottle','click_bell'],1)
for g,cp in zip([2,3],configs):
 c=json.loads(Path(cp).read_text());c['gpu']=g;Path(cp).write_text(json.dumps(c,indent=2))
 e=dict(base,REPO_PATH=str(R),RLINF_CODE_WORKING_DIR=str(R),EMBODIED_PATH=str(R/'examples/embodiment'),PYTHONPATH=str(D/'bootstrap')+':'+str(R)+':'+base['ROBOTWIN_PATH'],TMPDIR=str(D/f'tmp-g{g}'),CUDA_VISIBLE_DEVICES=str(g),DOJO_GPU_SCOPE=str(g));Path(e['TMPDIR']).mkdir();(D/f'environment-g{g}.json').write_text(json.dumps(e,indent=2))
(D/'plan.json').write_text(json.dumps({'time':time.time(),'gpus':[2,3],'configs':configs,'source':str(R),'training':'6 then full resume to 8, native N1 export check','batch':384,'micro_per_rank':192,'stop':'no formal training; collection timeout 1800s, each train stage 900s; exact owned process cleanup','authorized':'user conditional two free GPUs on 2026-10-26'},indent=2))
with (D/'driver.log').open('x')as f:
 p=subprocess.Popen([sys.executable,'-u','-B','/data/chenyiteng/fql1026_smoke_driver.py'],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'deployment':str(D),'gpus':[2,3],'plan':str(D/'plan.json')}))
