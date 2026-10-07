import collections,datetime,json,math,statistics
from pathlib import Path
B=Path('E:/Codex/home/visualizations/2026/10/02/01a0fc88-6d9d-7131-932a-4b2852e69b58');P=B/'expo-review-20261007';P.mkdir(exist_ok=True)
d=json.loads((B/'gpu0-fix-20261003/sz2/expo-review-20261007-v1.out').read_text());extra=json.loads((B/'gpu0-fix-20261003/sz2/expo-review-20261007-detail.out').read_text())
events=[e for e in extra['events'] if e['time']<=d['time']]
eps=[e for e in d['events'] if e['event']=='episode_finished'];learn=[e for e in d['events'] if e['event']=='learner_finished']
assert len(eps)==d['checkpoint']['cadence']['counters']['episodes_completed']==196
assert sum(e['physical_steps'] for e in eps)==28227 and len(learn)==655
assert all(e['call']==i for i,e in enumerate(learn,1))
assert all(math.isfinite(v) for e in learn for v in e['metrics'].values() if isinstance(v,(int,float)))
assert all(e['metrics']['base/base_frozen_parameters_with_grad']==0 for e in learn)
post=[e for e in learn if e['call']>=596]
assert all(e['metrics']['base/base_parallel_devices']==2 for e in post)
rows=[];wins=steps=0
for i,e in enumerate(eps,1):
 wins+=e['success'];steps+=e['physical_steps'];row=dict(episode=i,success=int(e['success']),actions=e['physical_steps'],cumulative_actions=steps,time=e['time'],seed=e['env_seed'])
 for n in (5,10,15,20):row['ma'+str(n)]=sum(v['success'] for v in eps[i-n:i])/n if i>=n else None
 rows.append(row)
ev=[]
for k,v in d['evaluations'].items():
 ep=0 if k=='initial-base' else 128 if k=='final-expo' else int(k.split('-')[-1])
 ev.append(dict(label=k,episode=ep,successes=v['successes'],n=v['episodes'],rate=v['success_rate'],seconds=v['elapsed_seconds'],calls=v['contract']['learner_calls'],rows=v['rows'],contract=v['contract']))
ev.sort(key=lambda v:v['episode']);assert len({v['contract']['seed_sha256'] for v in ev})==1 and all(v['n']==20 for v in ev)
latest=ev[-1];base=ev[0];end20=next(v for v in ev if v['episode']==128);best=next(v for v in ev if v['episode']==160)
def paired(a,b):
 aa={v['seed']:v['success_once'] for v in a['rows']};bb={v['seed']:v['success_once'] for v in b['rows']}
 assert aa.keys()==bb.keys()
 return dict(gained=[k for k in aa if not aa[k] and bb[k]],lost=[k for k in aa if aa[k] and not bb[k]],both_success=sum(aa[k] and bb[k] for k in aa))
recent=ev[-3:];persistent_fail=[r['seed'] for r in latest['rows'] if all(not next(x['success_once'] for x in v['rows'] if x['seed']==r['seed']) for v in recent)]
idx=d['replay_index']['online_entries'];succ=[e for e in idx if e['success']]
chunks=[e for e in events if e['event']=='real_chunk'];assert len(chunks)==d['event_counts']['real_chunk']
def selection(lo,hi):
 z=[e for e in chunks if lo<=e['episode']+1<=hi]
 return dict(chunks=len(z),edited=sum(e['selected']>=8 for e in z),edited_fraction=sum(e['selected']>=8 for e in z)/len(z))
fm=[e for e in events if e['event']=='base_fm_source'];lastfm=fm[-20:]
fm_share=sum(e['sources'].get('online',0) for e in lastfm)/(64*len(lastfm))
recent_short=[e for e in idx[-20:] if e['success'] and e['frames']<50]

windows=[]
for lo,hi in ((570,595),(596,615),(636,655)):
 rr=[e for e in learn if lo<=e['call']<=hi];windows.append(dict(calls=[lo,hi],devices=rr[0]['metrics']['base/base_parallel_devices'],mean_seconds=statistics.mean(e['learner_seconds'] for e in rr),gpu_seconds=statistics.mean(e['learner_seconds']*e['metrics']['base/base_parallel_devices'] for e in rr)))
# Complete episode-start interval excludes the current unfinished learner.
starts=[e for e in events if e['event']=='episode_started'];begin=next(e for e in starts if e['episode']==175)['time'];end=next(e for e in starts if e['episode']==195)['time']
span=end-begin;chosen=[e for e in events if begin<=e['time']<end]
learn_sec=sum(e['learner_seconds'] for e in chosen if e['event']=='learner_finished')
eval_sec=sum(d['evaluations'][e['label']]['elapsed_seconds'] for e in chosen if e['event']=='evaluation_finished')
collect_sec=0
for ep in range(175,195):
 s=next(e for e in starts if e['episode']==ep);f=next(e for e in eps if e['episode']==ep);collect_sec+=f['time']-s['time']
actions_span=sum(e['physical_steps'] for e in eps if 175<=e['episode']<195)
parts=dict(learning=learn_sec,evaluation=eval_sec,collection=collect_sec,save_reset_other=span-learn_sec-eval_sec-collect_sec)
remaining=60000-d['checkpoint']['cadence']['counters']['physical_actions']
eta_hours=remaining/(actions_span/span)*1/3600
signals={}
for name in ('critic_loss','critic_grad_norm','q_mean','target_mean','base/base_fm_loss','base/base_grad_norm','editor_grad_norm','temperature','entropy','edit_norm'):
 vals=[e['metrics'][name] for e in learn[-20:]];signals[name]={'last':vals[-1],'last20_mean':statistics.mean(vals),'last20_min':min(vals),'last20_max':max(vals)}
summary=dict(time=d['time'],beijing=d['beijing'],counters=d['checkpoint']['cadence']['counters'],owner_pid=3008973,driver_pid=d['driver_identity']['pid'],heartbeat_seconds=d['heartbeat_age_seconds'],source=d['source_commit'],source_files=len(d['source_manifest']),source_match=d['all_source_files_match'],errors=d['error_files'],online=dict(total_wins=wins,total_episodes=len(eps),last={str(n):sum(e['success'] for e in eps[-n:]) for n in (5,10,15,20,50)},successful_length_mean=statistics.mean(e['physical_steps'] for e in eps if e['success']),recent_successful_length_mean=statistics.mean(e['physical_steps'] for e in eps[-20:] if e['success'])),evaluations=ev,pairs=dict(initial_to_latest=paired(base,latest),end20k_to_latest=paired(end20,latest),best160_to_latest=paired(best,latest)),persistent_fail_last3=persistent_fail,learning_windows=windows,signals=signals,post_switch_calls=len(post),timing=dict(start=begin,end=end,wall_hours=span/3600,actions=actions_span,actions_per_hour=actions_span/span*3600,parts_seconds=parts,parts_percent={k:100*v/span for k,v in parts.items()},eta_hours_constant_recent_rate=eta_hours),replay=dict(online_episodes=len(idx),online_bytes=d['inventory']['replay']['bytes'],q_windows=sum(e['q_count'] for e in idx),fm_windows=sum(e['fm_count'] for e in idx),successes=len(succ),success_excluded_fm=sum(e['fm_count']==0 for e in succ),last20_success_excluded_fm=len(recent_short),last20_fm_online_share=fm_share,file_identity=extra['replay_file_identity']),edited_selection=dict(all=selection(1,196),last20=selection(177,196)),resources={k:d[k] for k in ('gpus','process_memory','memory','disk')},inventory=d['inventory'],checkpoints=d['checkpoints'],rlt_wait_binding_validation_missing='rlt-return-validation.json' not in d['control_files']['/data/chenyiteng/deployment-20261006/rlt-q-signals-g67-v1/coexist'])
for name,obj in [('summary',summary),('episodes',rows),('learning',learn),('snapshot',d),('detail',extra)]: (P/(name+'.json')).write_bytes((json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps({k:summary[k] for k in ('online','pairs','persistent_fail_last3','learning_windows','timing','replay','edited_selection','signals','rlt_wait_binding_validation_missing')},ensure_ascii=False,indent=2))
