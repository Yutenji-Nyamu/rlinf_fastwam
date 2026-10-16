import os,json,sys,time,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
T=Path('/data/chenyiteng/deployment-20261016/expo-ubc-b64-tau123-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,same,proc
def optional(p):
 try:return read(p)
 except (OSError,ValueError):return None
def tail(p,n=5000):
 if not p.exists():return ''
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
plan=read(S/'plan.json');state=read(S/'status.json');before=read(T/'prepared.json')['plan_before']
out=dict(time=time.time(),cutover=optional(T/'cutover-complete.json'),cutover_error=optional(T/'cutover-error.json'),owner=read(S/'owner-identity.json'),owner_alive=same(read(S/'owner-identity.json')),queue={},groups=[],gpu=[])
for g,slot in plan['slots'].items():
 f=read(slot['fallback']);rt=Path(f['runtime']);cp=Path(f['checkpoint'])
 out['queue'][g]=dict(phase=state['slots'][g]['phase'],external_alive=same(slot['external_identity']),fallback_unchanged=slot['fallback']==before['slots'][g]['fallback'],checkpoint_complete=bool((optional(cp/'actor/sac_components/rlt_trainer_state/complete.json') or {}).get('complete')),config=(rt/'resolved.yaml').exists(),environment=(rt/'environment.json').exists(),not_dispatched=not (rt/'started.json').exists(),status=state['slots'][g])
for p in read(T/'prepared.json')['groups']:
 q=read(p['request']);rt=Path(q['runtime']);run=Path(q['run'])/'run';ident=optional(rt/'driver.json');events=[]
 for line in tail(run/'events.jsonl',30000).splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
 ep=[]
 for f in (run/'evaluations').glob('*/attempt-*/events.jsonl'):
  rows=[]
  for line in tail(f,18000).splitlines():
   try:rows.append(json.loads(line))
   except ValueError:pass
  ep.append({'path':str(f),'data':rows[-5:]})
 out['groups'].append(dict(tau=q['tau'],gpus=q['gpus'],request=q['request'],run=q['run'],driver=ident,alive=bool(ident and same(ident)),released=optional(rt/'released.json'),old_alive=same(p['old_identity']),status=optional(run/'status.json'),checkpoint=optional(run/'checkpoint.json'),events=events[-8:],driver_tail=tail(rt/'driver.log'),pair_owner=optional(rt/'owner.json'),pair_status=optional(rt/'status.json'),roster=optional(rt/'roster.json'),failed=optional(run/'failed.json'),evaluation_progress=ep))
for g,v in before['slots'].items():
 if all(int(g) not in p['gpus'] for p in read(T/'prepared.json')['groups']):assert v==plan['slots'][g] and same(v['external_identity'])
x=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],timeout=25))
for i,g in enumerate(x.findall('gpu')):
 processes=[]
 for r in g.findall('processes/process_info'):
  pid=int(r.findtext('pid'));ident=proc(pid);e={}
  if ident and ident['uid']==os.getuid():
   try:e=dict(v.split('=',1) for v in Path(f'/proc/{pid}/environ').read_bytes().decode(errors='replace').split('\0') if '=' in v)
   except OSError:pass
  processes.append(dict(type=r.findtext('type'),used=r.findtext('used_memory'),identity=ident,binding={k:e.get(k) for k in ['CUDA_VISIBLE_DEVICES','EGL_DEVICE_ID','EXPO_SMOKE_OWNER_SCOPE','CLUSTER_NAMESPACE']}))
 out['gpu'].append(dict(index=i,uuid=g.findtext('uuid'),used=g.findtext('fb_memory_usage/used'),util=g.findtext('utilization/gpu_util'),processes=processes))
print(json.dumps(out))
