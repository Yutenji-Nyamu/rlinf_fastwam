"""Read only compact diagnostic progress and final GPU binding."""
import datetime,json,os,socket,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-diagnosis-v1');O=D/'run'
def read(p):return json.loads(p.read_text()) if p.is_file() else None
r=read(O/'result.json') or {}
out={'time':datetime.datetime.now().astimezone().isoformat(),'launch':read(D/'launch-receipt.json'),
     'heartbeat':read(O/'heartbeat.json'),'final':read(O/'final.json'),
     'engineering_passed':r.get('engineering_passed'),'error':r.get('error'),
     'elapsed_s':r.get('elapsed_s'),'cases':[], 'throughput':[]}
for case in r.get('cases',[]):
    p=case.get('preprocessing',{});o=case.get('output',{})
    out['cases'].append({'id':case['case']['id'],'status':case['status'],'error':case.get('error'),
        'frames':p.get('frames'),'success':o.get('success'),'match':o.get('match'),'raw_text':o.get('raw_text'),
        'parse_status':o.get('parse_status'),'input_tokens':p.get('input_tokens'),
        'peaks_gib':[b.get('peak_allocated_bytes',0)/1024**3 for b in case.get('batches',[])]})
for row in r.get('throughput',[]):out['throughput'].append({k:v for k,v in row.items() if k not in ('outputs','batches')})
for p in (D/'owner-launch.log',O/'diagnostic.log'):
    if p.exists():out[p.name]=p.read_text(errors='replace')[-2500:]
xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
out['gpus']=[]
for i,g in enumerate(xml.findall('gpu')):
    out['gpus'].append({'index':i,'used':g.findtext('./fb_memory_usage/used'),'util':g.findtext('./utilization/gpu_util'),
        'processes':[{'pid':p.findtext('pid'),'type':p.findtext('type'),'memory':p.findtext('used_memory')} for p in g.findall('./processes/process_info')]})
print(json.dumps(out,ensure_ascii=False))
