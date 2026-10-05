"""Read-only compact experiment receipt suitable for Git; no environment dump."""
import datetime,hashlib,json,os,socket,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D=S/'rynn-control-v2';O=S/'runs/rynn-success-v2';OLD=S/'runs/rynn-success-v1'
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compact(value,keys):return {key:value[key] for key in keys if key in value} if value else None
plan=read(O/'owner-plan.json') or read(D/'prepared/plan.json')
tests=read(D/'prepared/cpu-tests.json')
gate=read(O/'rm_gate/result.json')
out={'time':datetime.datetime.now().astimezone().isoformat(),'host':socket.gethostname(),'uid':os.getuid(),
     'owner_dir':str(O),'control_dir':str(D),'plan_sha256':sha(D/'prepared/plan.json'),
     'owner_identity':read(O/'owner-identity.json'),'phase':read(O/'state.json'),
     'cpu_tests':{'passed':tests['all_cpu_tests_passed'],'exit_codes':[r['exit_code'] for r in tests['checks']],
                  'receipt_sha256':sha(D/'prepared/cpu-tests.json')},
     'config':plan['budget'],'current_error':read(O/'error.json'),
     'formal_launched':(O/'formal/driver-identity.json').exists(),
     'checkpoint_claim':'Continuation from old-RM CP70, not a fresh-SFT method comparison',
     'scope':'Rynn proxy and smoke metrics do not establish native task improvement'}
old_gate=read(OLD/'rm_gate/result.json');old_final=read(OLD/'final.json')
out['v1_failure']={'error':old_gate['error'],'service_error':next((x.get('last_error') for x in old_gate['offload'] if x.get('last_error')),None),
    'final':compact(old_final,['time','terminal_status','recovery_error','rlt_return_dispatched']),
    'cleanup_all_stopped':read(OLD/'cleanup.json')['all_stopped']}
if gate:
    out['rm_gate']=compact(gate,['time','passed','quality_scope','benchmarks_use_repeated_inputs','selected_rm_batch',
        'measurements','sanity','error','b16_skipped','sample_sha256'])
    out['rm_gate']['offload']=[compact(r,['ok','is_offloaded','physical_gpu','last_error','error','memory']) for r in gate['offload']]
    out['rm_gate']['sha256']=sha(O/'rm_gate/result.json')
for name in ['batch16-smoke-gate.json','rynn-learning-smoke-gate.json','startup_smoke/result.json','startup_smoke/driver-finished.json',
             'formal/driver-identity.json','formal/driver-finished.json','final.json']:
    value=read(O/name)
    if value is not None:out[name]=value
xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
out['gpus']=[]
for index,gpu in enumerate(xml.findall('gpu')):
    rows=[]
    for process in gpu.findall('./processes/process_info'):
        pid=int(process.findtext('pid'));proc=Path('/proc')/str(pid)
        try:uid=proc.stat().st_uid
        except FileNotFoundError:uid=None
        rows.append({'pid':pid,'uid':uid,'type':process.findtext('type'),'used_memory':process.findtext('used_memory')})
    out['gpus'].append({'index':index,'memory_used':gpu.findtext('./fb_memory_usage/used'),
        'utilization':gpu.findtext('./utilization/gpu_util'),'processes':rows})
out['owned_contexts_outside_scope']=[dict(gpu=gpu['index'],**r) for gpu in out['gpus'] if gpu['index'] not in (4,5,6,7)
    for r in gpu['processes'] if r['uid']==20001]
print(json.dumps(out,ensure_ascii=False))
