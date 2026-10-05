"""Compact diagnostic and exact return receipts; excludes media and environments."""
import datetime,hashlib,json,os,socket
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-diagnosis-v1');O=D/'run'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=read(O/'result.json');f=read(O/'final.json');tests=read(D/'prepared/cpu-tests.json')
out={'time':datetime.datetime.now().astimezone().isoformat(),'host':socket.gethostname(),'uid':os.getuid(),
    'owner_dir':str(O),'owner_identity':read(O/'owner-identity.json'),
    'plan_sha256':sha(D/'prepared/plan.json'),'result_sha256':sha(O/'result.json'),
    'cpu_tests':{'passed':tests['all_cpu_tests_passed'],'receipt_sha256':sha(D/'prepared/cpu-tests.json'),
        'exit_codes':[row['exit_code'] for row in tests['checks']]},
    'engineering_passed':r['engineering_passed'],'quality_passed':r['quality_passed'],
    'error':r.get('error'),'elapsed_s':r.get('elapsed_s'),'cases':[], 'throughput':[],
    'sources':r['sources'],'final':f,'cleanup_all_stopped':read(O/'cleanup.json')['all_stopped'],
    'scope':'Input diagnosis and repeated-input speed only; no policy training or native improvement claim.'}
for row in r['cases']:
    p=row.get('preprocessing',{});o=row.get('output',{})
    out['cases'].append({'id':row['case']['id'],'status':row['status'],'description':row['case'].get('description'),
        'reference_label':row['case'].get('expected_reference'),'reference_label_note':'Expert provenance only, not a new physics check',
        'frames':p.get('frames'),'input_tokens':p.get('input_tokens'),'distinct_processed_frames':p.get('distinct_processed_frames'),
        'success':o.get('success'),'match':o.get('match'),'raw_text':o.get('raw_text'),'parse_status':o.get('parse_status'),
        'error':row.get('error'),'peak_allocated_gib':max([b.get('peak_allocated_bytes',0)/1024**3 for b in row.get('batches',[])] or [0])})
for row in r.get('throughput',[]):out['throughput'].append({k:v for k,v in row.items() if k not in ('outputs','batches')})
out['throughput_stopped']=r.get('throughput_stopped')
out['offload']={k:v for k,v in (r.get('offload') or {}).items() if k in ('ok','is_offloaded','physical_gpu','last_error','memory')}
print(json.dumps(out,ensure_ascii=False))
