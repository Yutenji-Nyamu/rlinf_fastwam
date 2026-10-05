"""Read-only native-input diagnostic progress and compact publishable results."""
import datetime,hashlib,json,os,socket,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-diagnosis-v2');O=D/'run'
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
r=read(O/'result.json') or {};tests=read(D/'prepared/cpu-tests.json') or {}
out={'time':datetime.datetime.now().astimezone().isoformat(),'host':socket.gethostname(),'uid':os.getuid(),
    'owner_dir':str(O),'owner_identity':read(O/'owner-identity.json'),'heartbeat':read(O/'heartbeat.json'),
    'final':read(O/'final.json'),'result_sha256':sha(O/'result.json'),'plan_sha256':sha(D/'prepared/plan.json'),
    'engineering_passed':r.get('engineering_passed'),'quality_passed':r.get('quality_passed'),
    'error':r.get('error'),'elapsed_s':r.get('elapsed_s'),'cases':r.get('cases',[]),'sources':r.get('sources',{}),
    'cpu_tests_passed':tests.get('all_cpu_tests_passed'),'cpu_tests_sha256':sha(D/'prepared/cpu-tests.json'),
    'cleanup':read(O/'cleanup.json'),'offload':{k:v for k,v in r.get('offload',{}).items() if k in ('ok','is_offloaded','physical_gpu','memory')},
    'scope':'Four K8 B1 native-input comparisons, no policy training and no native improvement claim.'}
if not out['final']:
    for p in (D/'owner-launch.log',O/'diagnostic.log'):
        if p.exists():out[p.name]=p.read_text(errors='replace')[-2500:]
xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
out['gpus']=[{'index':i,'used':g.findtext('./fb_memory_usage/used'),'util':g.findtext('./utilization/gpu_util'),
    'processes':[{'pid':p.findtext('pid'),'type':p.findtext('type'),'memory':p.findtext('used_memory')} for p in g.findall('./processes/process_info')]} for i,g in enumerate(xml.findall('gpu'))]
print(json.dumps(out,ensure_ascii=False))
