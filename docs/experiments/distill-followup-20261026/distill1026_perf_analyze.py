import json,statistics
from pathlib import Path
E=Path('E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a');x=json.loads((E/'endpoint1023/distill1026-followup-live.out').read_text());out={}
for method,v in x['training'].items():
 events=v['queue_events'];metrics=v['metrics'];rows=[];prev=0
 for i,e in enumerate(events):
  if e['phase']!='TRAINING':continue
  stop=e['stop_step'];mm=[m for m in metrics if prev<m['step']<=stop];end=events[i+1]['time']if i+1<len(events)else x['time'];total=end-e['time']
  assert mm
  startup=mm[0]['time']-e['time']-(mm[0]['step']-prev)*mm[0]['seconds_per_update']
  intervals=[(b['time']-a['time'])/(b['step']-a['step'])for a,b in zip(mm,mm[1:])if b['step']>a['step']]
  med=statistics.median(intervals)if intervals else mm[-1]['seconds_per_update']
  update=(mm[-1]['step']-prev)*med
  rows.append(dict(start=prev,end=mm[-1]['step'],planned_stop=stop,total=total,startup_estimate=startup,median_update=med,update_estimate=update,save_wait_residual=total-startup-update))
  prev=stop
 ev=[s for s in v['summaries'].values()if s['condition'].startswith(method+'-s')]
 out[method]=dict(segments=rows,total_train_phase_s=sum(r['total']for r in rows),startup_s=sum(r['startup_estimate']for r in rows),updates_estimate_s=sum(r['update_estimate']for r in rows),save_wait_residual_s=sum(r['save_wait_residual']for r in rows),evaluation_s=sum(s['elapsed_seconds']for s in ev),eval_range_s=[min(s['elapsed_seconds']for s in ev),max(s['elapsed_seconds']for s in ev)],eval_count=len(ev),step=metrics[-1]['step'],checkpoint_gib=[(p['name'],round(p['bytes']/2**30,2))for p in v['checkpoints'][-4:]])
(E/'dmd-audit1026/followup-performance.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
