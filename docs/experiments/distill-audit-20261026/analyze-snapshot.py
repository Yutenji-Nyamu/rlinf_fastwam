import tarfile,json,statistics,math,collections,csv
from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--archive',type=Path,default=Path(__file__).with_name('recent-light-snapshot.tar.gz'))
parser.add_argument('--output',type=Path,default=Path('.'))
parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[3])
args=parser.parse_args();E=args.output;E.mkdir(parents=True,exist_ok=True);R=args.source_root
with tarfile.open(args.archive)as tar:
 blobs={m.name:tar.extractfile(m).read()for m in tar if m.isfile()}
def read(n):return json.loads(blobs[n])
def root(m):return f'results/pi05-distill/{m}-multitask-c10-sz1-g'+('45'if m=='endpoint'else'67')+'-20261025'
def episodes(s):
 out={};missing=[]
 for row in s['tasks']:
  if not row['valid']:continue
  path=row['output'].replace('/data/chenyiteng/','')+'/episodes.json'
  if path not in blobs:missing.append(path);continue
  for ep in read(path):
   out[row['task'],ep['requested']]=ep
 return out,missing
summaries={}
for m in ['endpoint','onedp']:
 for n in blobs:
  if n.startswith(root(m)+'/summary-')and 'smoke'not in n:summaries[m+':'+n.split('/summary-')[1][:-5]]=read(n)
raw=read(root('endpoint')+'/summary-raw-n1.json');n10=read(root('endpoint')+'/summary-raw-n10.json')
ref,miss=episodes(raw);points=[];pairs={};taskdata={}
for label,s in summaries.items():
 if ':raw' in label:continue
 eps,em=episodes(s);common=set(ref)&set(eps)
 if em or miss:print('missing episode records',em,miss)
 rows={t['task']:t for t in s['tasks']if t['valid']}
 a={t['task']:t for t in raw['tasks']if t['valid']}
 pairs[label]={'episode_pairs':len(common),'new_success':sum(not ref[k]['success']and eps[k]['success']for k in common),'lost_success':sum(ref[k]['success']and not eps[k]['success']for k in common),'different_actual_seed':sum(ref[k]['actual']!=eps[k]['actual']for k in common),'native_retries':sum(e['native_retry']for e in eps.values()),'unique_task_actual_seeds':len(set((k[0],v['actual'])for k,v in eps.items())),'tasks_up':sum(rows[k]['successes']>a[k]['successes']for k in rows),'tasks_down':sum(rows[k]['successes']<a[k]['successes']for k in rows)}
 if label.endswith('005565'):taskdata[label]=[{ 'task':k,'raw_n1':a[k]['successes'],'student':rows[k]['successes'],'delta':rows[k]['successes']-a[k]['successes']}for k in sorted(rows)]
 points.append({'method':label.split(':')[0],'step':int(s['condition'].split('-s')[1]),'episodes':s['episodes'],'successes':s['successes'],'rate':s['mean_success_rate'],'tasks':s['valid_tasks']})
metrics={};stats={};source={}
for m in ['endpoint','onedp']:
 rows=[json.loads(l)for l in blobs[root(m)+f'/{m}/metrics-rank0.jsonl'].decode().splitlines()];metrics[m]=rows
 keys=['loss','endpoint_mse','fake','score_gradient_rms','grad_norm','fake_grad_norm']
 bins={}
 for lo,hi in [(1,100),(101,556),(557,1113),(1114,2226),(2227,3339),(3340,5565),(5566,100000)]:
  subset=[r for r in rows if lo<=r['step']<=hi]
  if subset:bins[f'{lo}-{min(hi,rows[-1]["step"])}']={'n':len(subset),**{k:statistics.median(r[k]for r in subset)for k in keys if k in subset[0]}}
 stats[m]={'last':rows[-1],'bins':bins,'nonfinite':sum(not math.isfinite(v)for r in rows for v in r.values()if isinstance(v,(float,int))),'grad_norm_max':max(r['grad_norm']for r in rows),'clipped_logged_steps':sum(r['grad_norm']>1 for r in rows)}
 if m=='onedp':stats[m]['fake_grad_norm_max']=max(r['fake_grad_norm']for r in rows)
 source[m]={}
 for rel in ['rlinf/algorithms/pi05_distillation/losses.py','rlinf/algorithms/pi05_distillation/data.py','tools/pi05_distillation/train.py']:
  source[m][rel]=blobs[f'deployed-source/{m}/{rel}'].decode().replace('\r\n','\n')==(R/rel).read_text().replace('\r\n','\n')
validsets=[set(r['task']for r in s['tasks']if r['valid'])for s in summaries.values()]
report={'snapshot':read('MANIFEST.json')['time'],'raw_n1':{'successes':raw['successes'],'episodes':raw['episodes']},'raw_n10':{'successes':n10['successes'],'episodes':n10['episodes']},'same_valid_tasks_every_point':all(x==validsets[0]for x in validsets),'missing_tasks':[r['task']for r in raw['tasks']if not r['valid']],'points':sorted(points,key=lambda r:(r['method'],r['step'])),'paired_vs_n1':pairs,'task_deltas_5565':taskdata,'training':stats,'deployed_source_matches':source}
(E/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf8')
for m,rows in metrics.items():(E/(m+'-metrics.jsonl')).write_text(''.join(json.dumps(r)+'\n'for r in rows),encoding='utf8')
print(json.dumps({'sources_match':source,'same_valid_tasks':report['same_valid_tasks_every_point'],'missing_tasks':report['missing_tasks'],'final_paired':{k:v for k,v in pairs.items()if k.endswith('005565')},'training':stats},indent=2))
for label,rows in taskdata.items():print(label,'best',sorted(rows,key=lambda r:r['delta'],reverse=True)[:6],'worst',sorted(rows,key=lambda r:r['delta'])[:6])
