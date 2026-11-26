from pathlib import Path as P
from collections import Counter
from statistics import mean,median
import json,math
d=P(__file__).resolve().parent
def lines(name):return [json.loads(t) for t in (d/name).read_text(encoding='utf-8-sig').splitlines() if t.strip()]
def write(name,v):(d/name).write_text(json.dumps(v,ensure_ascii=False,indent=1),encoding='utf8')
allw={g:lines(f'progress-g{g}.jsonl') for g in (5,6)}
ev={g:[w['rows'] for w in v if w['mode']=='eval'] for g,v in allw.items()}; tr={g:[w['rows'] for w in v if w['mode']=='train'] for g,v in allw.items()}
evlogs=lines('progress-evals.jsonl');el={}
for i,item in enumerate(evlogs):
 rows=ev[5][i]+ev[6][i];assert len(rows)==len({r['layout'] for r in rows})==32
 expected=item.get('successes') if 'successes' in item else round(item['metrics']['eval/success_once']*32)
 assert sum(r['success'] for r in rows)==expected
 el[item['round']]=rows
n=min(len(tr[5]),len(tr[6]))//2
train={r:sum([wave for g in (5,6) for wave in tr[g][(r-1)*2:r*2]],[]) for r in range(1,n+1)}
def measure(rows):
 assert all(r['bottles'] in range(5) for r in rows)
 assert all(not r['success'] or r['bottles']==4 for r in rows)
 counts=Counter(r['bottles'] for r in rows);total=len(rows)
 wins=sum(r['success'] for r in rows);four_no=counts[4]-wins
 # Log milestones represent a running maximum and are completed to four at binary success.
 # Dojo evaluation reports 100 for finished successes; failed trials are capped at 40.
 official_rules=[100 if r['success'] else (0,10,25,40,40)[r['bottles']] for r in rows]
 potentially_imputed=[r for r in rows if r['success'] and r['first_bottle_steps'][3]>=r['actions']]
 return dict(episodes=total,counts=[counts[i] for i in range(5)],reached=[sum(counts[k] for k in range(i,5)) for i in range(1,5)],successes=wins,success_rate=wins/total,mean_bottles=mean(r['bottles'] for r in rows),mean_official_rubric=mean(official_rules),mean_failed_bottles=mean([r['bottles'] for r in rows if not r['success']]) if wins<total else None,mean_linear_reward=mean(r['bottles']/4 for r in rows),four_without_success=four_no,success_fourth_recorded_at_last_action=len(potentially_imputed))
evstats={k:measure(v) for k,v in el.items()};trstats={k:measure(v) for k,v in train.items()}
for rows in train.values():
 assert len(rows)==128 and all(abs(r['bottles']*.25-r['episode_return'])<1e-6 for r in rows)
for k,v in evstats.items():assert 0<=v['mean_official_rubric']<=100 and v['mean_bottles']==sum(v['reached'])/v['episodes']
e25={x['round']:measure(x['rows']) for x in lines('progress-standard.jsonl') if len(x.get('rows',[]))==25}
# The first eval lists all 32 layouts. If the public 25-layout receipt is absent use their actual ID subset.
if 0 not in e25:e25[0]=measure([r for r in el[0] if r['layout'] in set(range(25))])
for k,v in e25.items():
 assert v['episodes']==25 and v['counts']==measure([r for r in el[k] if r['layout'] in set(range(25))])['counts']
def groupstats(rows):return measure([r for sub in rows for r in sub])
summary={'first10':groupstats([train[k] for k in range(1,11)]),'last10':groupstats([train[k] for k in range(n-9,n+1)]),'all':groupstats(list(train.values())),'last3eval':groupstats([el[k] for k in sorted(el)[-3:]])}
# Paired comparison of actual layouts; report transitions for the two directly measured outcomes.
base={r['layout']:r for r in el[0]};latest={r['layout']:r for r in el[max(el)]};diff=[latest[i]['bottles']-base[i]['bottles'] for i in sorted(base)];paired=dict(more=sum(i>0 for i in diff),same=diff.count(0),fewer=sum(i<0 for i in diff),net=sum(diff),n=len(diff))
result=dict(eval32=evstats,eval25=dict(sorted(e25.items())),train128=trstats,periods=summary,paired_bottles=paired,latest_complete_training_round=n,score_definition='按 Dojo 档位逐局重建；训练回报为每瓶 0.25。逐局瓶数是达到过的最高档，成功时补至四瓶。',source=['https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/tasks/put_bottles_into_dustbin.py','https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/env/reward_manager/reward_manager.py'])
eval_rows={str(k):[[r['layout'],r['bottles'],int(r['success'])] for r in sorted(rows,key=lambda r:r['layout'])] for k,rows in el.items()}
write('bottle-eval-layouts.json',dict(columns=['layout','bottles','success'],rounds=eval_rows))
result['source'].append('https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/src/eval_client/eval_env.py')
write('bottle-progress.json',result)
print('EVAL',[(k,v['successes'],round(v['mean_bottles'],4),round(v['mean_official_rubric'],3),round(v['mean_failed_bottles'],2),v['success_fourth_recorded_at_last_action']) for k,v in evstats.items()])
print('train rounds',n,'first,last,all mean & official score & sr',[(k,round(summary[k]['mean_bottles'],4),round(summary[k]['mean_official_rubric'],4),round(summary[k]['success_rate'],5)) for k in summary], 'paired',paired)
print('25 eval',[(k,v['successes'],round(v['mean_bottles'],4),round(v['mean_official_rubric'],3)) for k,v in sorted(e25.items())])

