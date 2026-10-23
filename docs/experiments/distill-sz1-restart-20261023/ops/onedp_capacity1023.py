"""Targeted OneDP128/192 trials using the accepted train/resume/export path."""
import importlib.util,json,os,sys,time,signal,traceback,shutil
from pathlib import Path
D=Path(sys.argv[1]);F=D.parent/'onedp-multitask-sz1-g67';previous=D.parent/'onedp-multitask-smoke-sz1-g67'
R=Path(json.loads((D/'prepared.json').read_text())['source'])
spec=importlib.util.spec_from_file_location('q',R/'tools/pi05_distillation/run_queue.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
sys.path.insert(0,str(D.parent));from priority_pair_handoff import handoff
me=q.proc(os.getpid());q.save(D/'identity.json',me);owns=False
try:
 deadline=time.time()+3600
 while not(previous/'released.json').exists():
  assert time.time()<deadline,'Initial smoke has not released the pair';time.sleep(5)
 assert q.read(previous/'smoke-accepted.json')['passed']
 state_path=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/status.json')
 while any(q.read(state_path)['slots'][str(g)]['phase']not in ['RLT_RUNNING','RLT_EXITED','WAITING_RLT']for g in [6,7]):
  assert time.time()<deadline;time.sleep(2)
 handoff(D,[6,7],me);owns=True
 base=q.read(D/'onedp.json');results=[]
 oldO=Path(q.read(previous/'prepared.json')['output'])/'smoke-onedp'
 rows=[json.loads(s)for s in(oldO/'metrics-rank0.jsonl').read_text().splitlines()if s];rows=[r for r in rows if r['step']<=12]
 elapsed=(rows[-1]['time']-rows[1]['time'])/(rows[-1]['step']-rows[1]['step'])
 results.append(dict(micro=64,global_batch=128,oom=False,seconds_per_update=elapsed,windows_per_second=128/elapsed,peak_gib=max(r['peak_gib']for r in rows),peak_reserved_gib=max(r['peak_reserved_gib']for r in rows),source=str(previous)))
 for micro in [128,192]:
  label=f'onedp-m{micro}';c=dict(base,batch_size=micro*2,microbatch_per_rank=micro,log_every=1);q.save(D/f'{label}.json',c);out=q.O/f'smoke-{label}'
  try:receipt=q.train(label,out,6,label=f'capacity-{label}')
  except RuntimeError:
   log=(D/f'capacity-{label}.log').read_text(errors='replace')
   if 'out of memory'not in log.lower():raise
   deadline=time.time()+90
   while any(q.gpus()[g]['processes']for g in [6,7]):
    assert time.time()<deadline;time.sleep(1)
   result=dict(micro=micro,global_batch=micro*2,oom=True)
  else:
   rows=[json.loads(s)for s in(out/'metrics-rank0.jsonl').read_text().splitlines()];assert rows[-1]['step']==6
   sec=(rows[-1]['time']-rows[1]['time'])/(rows[-1]['step']-rows[1]['step'])
   result=dict(micro=micro,global_batch=micro*2,oom=False,seconds_per_update=sec,windows_per_second=micro*2/sec,peak_gib=max(r['peak_gib']for r in rows),peak_reserved_gib=max(r['peak_reserved_gib']for r in rows),checkpoint=receipt)
  results.append(result);q.save(D/'capacity-results.json',results)
  # The measured128 allocation predicts whether192 can fit80GiB, without a sweep.
  if micro==128 and(result['oom']or result['peak_gib']+(result['peak_gib']-results[0]['peak_gib'])>79):break
 candidates=sorted([r for r in results if not r['oom']],key=lambda r:r['windows_per_second'],reverse=True)
 selected=candidates[0];micro=selected['micro']
 if micro!=64:
  label=f'onedp-m{micro}';out=q.O/f'smoke-{label}';q.train(label,out,14,selected['checkpoint']['path'],label=f'capacity-{label}-resume14');q.accept_smoke(label)
  accepted=q.read(D/f'smoke-{label}-accepted.json')
 else:accepted=q.read(previous/'smoke-onedp-accepted.json')
 formal=q.read(F/'onedp.json');formal.update(batch_size=micro*2,microbatch_per_rank=micro);q.save(F/'onedp.json',formal)
 p=q.read(F/'prepared.json');p.update(training_global_batch=micro*2,per_gpu_microbatch=micro);q.save(F/'prepared.json',p)
 q.save(F/'smoke-onedp-accepted.json',accepted);q.save(F/'smoke-accepted.json',dict(time=time.time(),passed=True,capacity_source=str(D),micro=micro,selected=selected))
 schedule=q.read(F/'schedule.json');schedule.update(run_smoke_before_formal=False,capacity_source=str(D),smoke_source=str(D));q.save(F/'schedule.json',schedule)
 q.save(D/'selected.json',dict(time=time.time(),selected=selected,formal_epochs=20,reason='Highest measured samples per second among successful trials; same total sample budget'))
 q.event('CAPACITY_COMPLETE',selected=selected)
except BaseException:
 q.event('FAILED',error=traceback.format_exc());raise
finally:
 if owns:
  deadline=time.time()+90
  while any(q.gpus()[g]['processes']for g in [6,7])and time.time()<deadline:time.sleep(1)
  if all(not q.gpus()[g]['processes']for g in [6,7]):q.save(D/'released.json',dict(time=time.time(),identity=me,cards_clear=True,passed=(D/'selected.json').exists()))
