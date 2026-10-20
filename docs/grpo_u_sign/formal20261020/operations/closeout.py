"""Close the exact authorized EXPO run and protect its stopped final state."""
import csv,json,os,statistics,sys,tarfile,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'))
from common import read,save,same
D=Path(sys.argv[1]);R=Path(read(D/'handoff.json')['oldrun'])
assert not same(read(D/'handoff.json')['old_driver'])
release=read(D/'old-released.json');assert release['owned_contexts_clear']and release['cleanup']['ok']
assert not(D/'expo-closeout.json').exists()
cp=read(R/'run/checkpoint.json');assert cp['finite']
P=Path('/data/chenyiteng/models/protected-eval/turn_switch')/(R.name+'_20261020_final')
P.mkdir(parents=True,exist_ok=False);os.link(R/'run/checkpoint-latest.pt',P/'checkpoint.pt')
save(P/'checkpoint.json',cp);save(P/'protection.json',{'do_not_delete':True,'role':'closed_final_checkpoint','task':'turn_switch','source':str(R),'not_peak_claim':True})
ev=[{'evaluation':f.parent.name,**read(f)}for f in sorted((R/'run/evaluations').glob('*/complete.json'))]
rows=[r for r in ev if r['evaluation'].startswith('episode-')]
light=[{'evaluation':r['evaluation'],'successes':r['successes'],'episodes':r['episodes'],'success_rate':r['success_rate'],'updates':r['contract']['base_updates']}for r in ev]
summary={'time':time.time(),'method':R.name,'task':'turn_switch','run':str(R),
 'status':'user_authorized_early_closeout','budget_complete':False,'checkpoint':cp,'protected_checkpoint':str(P/'checkpoint.pt'),
 'initial':next(r['success_rate']for r in ev if r['evaluation']=='initial-base'),'latest':rows[-1]['success_rate'],
 'first5_mean':statistics.mean(r['success_rate']for r in rows[:5]),'last5_mean':statistics.mean(r['success_rate']for r in rows[-5:]),
 'peak_fixed20':max(r['success_rate']for r in rows),'evaluations':light,
 'interpretation':'Improvement over the raw initial model but variable fixed20 results. User elected early closeout; no full-budget, independent-seed, or matched-budget causal claim.'}
save(D/'expo-closeout.json',summary)
with(D/'evaluations.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(light[0]));w.writeheader();w.writerows(light)
files=[(R/n,n)for n in ['inputs.json','run/events.jsonl','run/status.json','run/stopped.json','run/resolved.json','run/checkpoint.json']]
files += [(f,str(f.relative_to(R)))for f in (R/'run/evaluations').glob('*/complete.json')]
log=R/'runtime/driver.log'
if log.stat().st_size<32*1024**2:files.append((log,'runtime/driver.log'))
else:
 with log.open('rb')as f:f.seek(-8*1024**2,2);(D/'driver-tail.log').write_bytes(f.read())
 files.append((D/'driver-tail.log','runtime/driver-tail.log'))
files.extend([(D/'expo-closeout.json','closeout.json'),(D/'evaluations.csv','evaluations.csv'),(D/'old-released.json','released.json')])
with tarfile.open(D/'expo-closeout.tar.gz','w:gz')as t:
 for p,name in files:
  if p.is_file():t.add(p,arcname=name,recursive=False)
print(json.dumps({'actions':cp['cadence']['counters']['physical_actions'],'latest':summary['latest'],'last5':summary['last5_mean'],'peak':summary['peak_fixed20'],'protected':str(P)}))
