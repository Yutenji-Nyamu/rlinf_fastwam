"""Archive the authorized stopped EXPO and protect its final complete snapshot."""
import csv,json,os,statistics,sys,tarfile,time
from pathlib import Path
D=Path('/data/chenyiteng/deployment-20261019/pi05-noise-early-c20-sz1-v1')
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'))
from common import read,save,same
p=read(D/'prepared.json');old=p['plan_before']['slots']['4'];assert not same(old['external_identity'])
release=read(old['external_release']);assert release['owned_contexts_clear']and release['cleanup']['ok']
R=Path(old['external_release']).parent.parent;cp=read(R/'run/checkpoint.json');assert cp['finite']
stopped=read(R/'run/stopped.json') if (R/'run/stopped.json').exists() else None
P=Path('/data/chenyiteng/models/protected-eval/turn_switch/expo_clean_b64_20261019_final')
assert not(D/'expo-closeout.json').exists()and not P.exists()
P.mkdir(parents=True)
os.link(R/'run/checkpoint-latest.pt',P/'checkpoint.pt')
save(P/'checkpoint.json',cp)
ev=[{'evaluation':f.parent.name,**read(f)}for f in sorted((R/'run/evaluations').glob('*/complete.json'))]
rows=[r for r in ev if r['evaluation'].startswith('episode-')]
light=[{'evaluation':r['evaluation'],'successes':r['successes'],'episodes':r['episodes'],
        'success_rate':r['success_rate'],'base_updates':r['contract']['base_updates'],
        'learner_calls':r['contract']['learner_calls']}for r in ev]
summary={'time':time.time(),'task':'turn_switch','method':'EXPO Clean B64 scaled entropy target',
 'run':str(R),'status':'user_authorized_early_closeout','reason':'User requests conclusion, archive and replacement with inference studies',
 'budget_complete':False,'checkpoint':cp,'protected_checkpoint':str(P/'checkpoint.pt'),
 'initial_success_rate':next(r['success_rate']for r in ev if r['evaluation']=='initial-base'),
 'latest_evaluation':light[-2] if light[-1]['evaluation']=='initial-base'else light[-1],
 'peak_fixed20':max(r['success_rate']for r in rows),'last5_mean':statistics.mean(r['success_rate']for r in rows[-5:]),
 'evaluations':light,'training_status':read(R/'run/status.json'),'stopped':stopped,
 'driver_exit_code':release['exit_code'],'interrupted_evaluation':'episode-000460',
 'stop_observation':'User SIGTERM was wrapped by native VectorEnv as RuntimeError; checkpoint1408 complete, evaluation460 incomplete, GPU cleanup confirmed.',
 'interpretation':'Same fixed20 training evaluation; repeated seeds, no independent confirmation or full-budget claim.'}
save(P/'protection.json',{'do_not_delete':True,'role':'closed_final_checkpoint','task':'turn_switch','source':str(R),'not_pillbottle_candidate':True})
save(D/'expo-closeout.json',summary)
with(D/'expo-clean-evaluations.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(light[0]));w.writeheader();w.writerows(light)
files=[(R/n,n)for n in ('inputs.json','run/events.jsonl','run/status.json','run/stopped.json','run/resolved.json','run/checkpoint.json')]
files += [(f,str(f.relative_to(R)))for f in (R/'run/evaluations').glob('*/complete.json')]
log=R/'runtime/driver.log'
if log.exists():
 if log.stat().st_size<16*1024**2:files.append((log,'runtime/driver.log'))
 else:
  with log.open('rb')as f:f.seek(-8*1024**2,2);(D/'expo-driver-tail.log').write_bytes(f.read())
  files.append((D/'expo-driver-tail.log','runtime/driver-tail.log'))
files += [(D/'expo-closeout.json','closeout.json'),(D/'expo-clean-evaluations.csv','evaluations.csv')]
a=D/'expo-clean-closeout.tar.gz'
with tarfile.open(a,'w:gz')as t:
 for f,name in files:
  if f.is_file():t.add(f,arcname=name,recursive=False)
save(D/'expo-archive.json',{'time':time.time(),'path':str(a),'bytes':a.stat().st_size,'protected':str(P)})
print(json.dumps({'latest':summary['latest_evaluation'],'last5':summary['last5_mean'],'peak':summary['peak_fixed20'],
 'actions':cp['cadence']['counters']['physical_actions'],'archive_bytes':a.stat().st_size,'protected':str(P)}))
