"""Local arithmetic over freshly collected timing records; no training execution."""
import hashlib,json,statistics,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'evidence/p069-budget-timing.out';data=json.loads(source.read_text());r=data['runs']
def mean(key,phase,metric):return r[key]['phases'][phase][metric]['mean']
bc_rows=r['bc_u']['step_rows'];bc_ordinary=statistics.mean(q['seconds']-q['eval'] for q in bc_rows)
bc_total=(bc_ordinary*400+mean('bc_u','bc','time/eval')*80+180)/3600
ratio=mean('rlt_u','precollection','time/generate_rollouts')/mean('rlt_dv','precollection','time/generate_rollouts')
pre=r['rlt_dv']['phases']['precollection']['time/step']['n'];warm=r['rlt_dv']['phases']['warmup']['time/step']['n'];online=2000-pre-warm
components={}
for phase,n in (('precollection',pre),('warmup',warm),('online',online)):
    v=r['rlt_dv']['phases'][phase]
    base=statistics.mean(q['seconds']-q['eval'] for q in r['rlt_dv']['step_rows'] if q['phase']==phase)
    delta=v['time/generate_rollouts']['mean']*(ratio-1)
    eval_sec=(v.get('time/eval') or r['rlt_dv']['phases']['online']['time/eval'])['mean']*ratio/25
    components[phase]={'rounds':n,'ordinary_seconds':base,'collection_overhead_seconds':delta,'eval_amortized_seconds':eval_sec,'hours':n*(base+delta+eval_sec)/3600}
rlt_total=sum(v['hours'] for v in components.values())+180/3600
now=time.time();start={'bc':datetime.fromisoformat('2026-10-06T13:01:45+08:00').timestamp(),'rlt':datetime.fromisoformat('2026-10-06T14:19:54+08:00').timestamp()}
out={'time':now,'timing_capture':data['time'],'timing_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'bc':{'total_steps':400,'ordinary_seconds':bc_ordinary,'eval_seconds':mean('bc_u','bc','time/eval'),'historical_mean_step':mean('bc_dv','bc','time/step'),
       'total_hours':bc_total,'remaining_hours':bc_total-(now-start['bc'])/3600,'total_hours_range':[35,42],'remaining_hours_range':[35-(now-start['bc'])/3600,42-(now-start['bc'])/3600]},
 'rlt':{'total_steps':2000,'collection_ratio_current_vs_history':ratio,'components':components,'total_hours':rlt_total,
       'remaining_hours':rlt_total-(now-start['rlt'])/3600,'total_hours_range':[52,68],'remaining_hours_range':[52-(now-start['rlt'])/3600,68-(now-start['rlt'])/3600]},
 'assumptions':'Same successful N4 pi0.5 historical pair; current U precollection cost scales historical collection/evaluation only; training timing transferred unchanged. Includes original warmup, eval/save inside step timers and 3min continuation overhead. Ranges are engineering sensitivity, not confidence intervals; shared load and future success can change speed.'}
for lane in ('bc','rlt'):
    out[lane]['central_finish_cst']=datetime.fromtimestamp(start[lane]+out[lane]['total_hours']*3600,timezone(timedelta(hours=8))).isoformat()
(ROOT/'evidence/budget-forecast.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
