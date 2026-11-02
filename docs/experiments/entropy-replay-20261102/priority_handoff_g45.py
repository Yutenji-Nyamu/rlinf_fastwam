"""Move the prepared paper replay to 4/5 at the next complete FQL eval.

Reuse accepted launchers and exact process identities. Keep the running 6/7
trainer untouched; remove only its idle replay/OT waiting owners.
"""
import fcntl,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal
F=Path('/home/chenyiteng/deployment-20261101/fql-v2/formal-g45')
E=Path('/home/chenyiteng/deployment-20261102/entropy-replay-v2')
T=Path('/home/chenyiteng/deployment-20261101/gad-ot-v1/formal-g67')
D=E.with_name('entropy-replay-g45-v3')
R=F.with_name('formal-g45-after-entropy-1102')
J=E.with_name('entropy-priority-1102')

def launch(command,directory,logname='queue.log'):
 with (directory/logname).open('a') as log:
  child=subprocess.Popen(command,cwd=directory,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 who=proc(child.pid)
 assert same(who)
 return who

def stop_idle(d,phasefile,phase):
 who=read(d/'identity.json')
 assert same(who) and read(d/phasefile)['phase']==phase
 exact_signal(who,signal.SIGTERM)
 deadline=time.time()+10
 while same(who):
  assert time.time()<deadline,'Idle owner did not exit'
  time.sleep(.1)
 return who

def main():
 assert os.getuid()==1003
 oldf=read(F/'identity.json');assert same(oldf)
 J.mkdir(exist_ok=False)
 save(J/'identity.json',proc(os.getpid()))
 assert read(E/'status.json')['phase']=='WAIT_CURRENT_TRAINING'
 assert read(T/'training-queue-status.json')['phase']=='PREPARED_WAIT_HANDOFF'
 p=read(E/'plan.json');gad=p['predecessor_identity'];assert same(gad)
 D.mkdir(exist_ok=False);shutil.copytree(E/'tools',D/'tools')
 shutil.copy2(E/'environment-template.json',D/'environment-template.json')
 p.update(gpus=[4,5],predecessor_identity=oldf)
 save(D/'plan.json',p)
 for i,g in enumerate([4,5]):
  save(D/f'replay-g{g}.json',{'gpu':g,'model_source':p['source'],'input_batch':p['input_batch'],'output':str(Path(p['output'])/'teacher4'),'queries':list(range(i,10,2)),'extra_batch':96})
 R.mkdir(exist_ok=False)
 for n in ['prepared.json','eval-template.json','fql_fixed_teacher.json','execution-source.json']:
  shutil.copy2(F/n,R/n)
 shutil.copytree(F/'seeds',R/'seeds')
 save(J/'status.json',{'phase':'READY_WAIT_FQL_EVAL55','time':time.time(),'old_fql':oldf,'new_replay':str(D),'new_fql':str(R)})
 # Move the idle replay out of 6/7 now; OT resumes its original predecessor.
 with (S/'priority-handoff.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  rp=read(S/'plan.json');tp=read(T/'formal-plan.json');tr=read(T/'registered.json')
  save(J/'before-rlt.json',rp);save(J/'before-ot-plan.json',tp)
  oldot=stop_idle(T,'training-queue-status.json','PREPARED_WAIT_HANDOFF')
  olde=stop_idle(E,'status.json','WAIT_CURRENT_TRAINING')
  (T/'registered.json').unlink()
  tp.update(predecessor_identity=gad,predecessor_deployment='/home/chenyiteng/deployment-20261101/gad-k4-v1/pair67')
  save(T/'formal-plan.json',tp)
  otwho=launch(['/usr/bin/python3','-u','-B',str(T.parent/'source/tools/pi05_distillation/single_formal_queue.py'),str(T)],T)
  tr.update(identity=otwho,predecessor_identity=gad,time=time.time(),insertion='paper replay moved to 4/5')
  save(T/'registered.json',tr)
  for g in ['6','7']:
   s=rp['slots'][g]
   s['external_identity']=otwho
   s['external_blockers']=[i for i in s.get('external_blockers',[]) if i not in [olde,oldot]]
   if gad not in s['external_blockers']:s['external_blockers'].append(gad)
  save(S/'plan.json',rp)
  save(E/'moved-to-g45.json',{'time':time.time(),'new_deployment':str(D),'old_identity':olde})
  save(J/'restored-67.json',{'identity':otwho,'predecessor':gad,'stopped_idle':[oldot,olde]})
 output=Path(read(F/'prepared.json')['output'])
 rowpath=output/'summary-fql_fixed_teacher-s000055.json'
 deadline=time.time()+3600
 while True:
  assert same(oldf),'Original FQL ended before requested boundary'
  if rowpath.exists():
   row=read(rowpath)
   if row.get('complete') and row.get('checkpoint',{}).get('step')==55:break
  if time.time()>deadline:raise TimeoutError('Eval55 did not finish within one hour; original training left running')
  time.sleep(.25)
 receipt=row['checkpoint']
 for k in ['path','policy','q']:assert Path(receipt[k]).is_file(),k
 rpplan=read(F/'formal-plan.json')
 rpplan.update(initial_receipts={'fql_fixed_teacher':receipt},resume_in_progress=['fql_fixed_teacher'])
 with (S/'priority-handoff.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert same(oldf)
  replaywho=launch(['/usr/bin/python3','-u','-B',str(D/'tools/entropy_queue.py'),str(D)],D)
  rpplan.update(predecessor_identity=replaywho,predecessor_deployment=str(D))
  save(R/'formal-plan.json',rpplan)
  fwho=launch(['/usr/bin/python3','-u','-B',str(F.parent/'source/tools/pi05_distillation/single_formal_queue.py'),str(R)],R)
  save(R/'registered.json',{'identity':fwho,'time':time.time(),'predecessor_identity':replaywho,'resumed_from':receipt,'order':'paper teacher K3 replay -> single-task FQL v2 remainder -> RLT'})
  rp=read(S/'plan.json')
  for g in ['4','5']:
   s=rp['slots'][g]
   s.update(external_identity=fwho,external_release=str(R/'released.json'))
   s['external_blockers']=s.get('external_blockers',[])+[oldf,replaywho]
  save(S/'plan.json',rp)
  save(D/'registered.json',{'identity':replaywho,'time':time.time(),'gpus':[4,5],'predecessor_identity':oldf,'successor_identity':fwho,'order':'paper replay -> single-task FQL v2 -> RLT'})
  save(J/'handoff.json',{'time':time.time(),'stopped_fql':oldf,'replay':replaywho,'resumed_fql':fwho,'resume':receipt,'evaluation':{'successes':row['successes'],'episodes':row['episodes']}})
  exact_signal(oldf,signal.SIGTERM)
 save(J/'status.json',{'phase':'HANDED_OFF','time':time.time(),'replay':replaywho,'resumed_fql':fwho,'resume_step':55})
 print(json.dumps(read(J/'status.json')),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  if J.exists():save(J/'error.json',{'time':time.time(),'error':traceback.format_exc()})
  raise
