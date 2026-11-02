import fcntl,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal
D=Path('/home/chenyiteng/deployment-20261102/grpo-clean-entropy-g45')
F=Path('/home/chenyiteng/deployment-20261101/fql-v2/formal-g45-after-entropy-1102')
R=F.with_name('formal-g45-after-grpo-1102')
assert os.getuid()==1003 and not R.exists() and not (D/'registered.json').exists()
old=read(F/'identity.json');assert same(old)
status=read(F/'training-queue-status.json');assert status['phase']=='TRAINING' and status['stop_step']==110,status
output=Path(read(F/'prepared.json')['output']);row=read(output/'summary-fql_fixed_teacher-s000055.json')
receipt=row['checkpoint'];assert row['complete'] and receipt['step']==55
assert read(output/'fql_fixed_teacher/checkpoint.json')['step']==55,'Next checkpoint completed; use new boundary'
for k in ['path','policy','q']:assert Path(receipt[k]).is_file()
R.mkdir()
for n in ['prepared.json','eval-template.json','fql_fixed_teacher.json','execution-source.json']:
 shutil.copy2(F/n,R/n)
shutil.copytree(F/'seeds',R/'seeds')
def launch(command,d):
 with (d/'queue.log').open('a') as log:
  c=subprocess.Popen(command,cwd=d,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 who=proc(c.pid);assert same(who);return who
with (S/'priority-handoff.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 p=read(D/'plan.json');p['predecessor_identity']=old;save(D/'plan.json',p)
 g=launch(['/usr/bin/python3','-u','-B',str(D/'tools/grpo_entropy_queue.py'),str(D)],D)
 fp=read(F/'formal-plan.json');fp.update(predecessor_identity=g,predecessor_deployment=str(D),initial_receipts={'fql_fixed_teacher':receipt},resume_in_progress=['fql_fixed_teacher'])
 save(R/'formal-plan.json',fp)
 f=launch(['/usr/bin/python3','-u','-B',str(F.parent/'source/tools/pi05_distillation/single_formal_queue.py'),str(R)],R)
 save(R/'registered.json',{'identity':f,'time':time.time(),'predecessor_identity':g,'resumed_from':receipt,'order':'Clean GRPO200 + final pool + six-point replay -> FQL v2 remainder -> RLT'})
 rp=read(S/'plan.json');save(D/'before-rlt.json',rp)
 for card in ['4','5']:
  s=rp['slots'][card];s.update(external_identity=f,external_release=str(R/'released.json'))
  s['external_blockers']=s.get('external_blockers',[])+[old,g]
 save(S/'plan.json',rp)
 rec={'identity':g,'time':time.time(),'gpus':[4,5],'predecessor_identity':old,'successor_identity':f,'fql_resume_step':55,'paper_replay_already_completed':True,'order':'Clean GRPO200 -> final-policy fixed pool32 -> six checkpoints K4 -> single FQL v2 -> RLT','protected_milestones':[0,40,80,120,160,200]}
 save(D/'registered.json',rec)
 exact_signal(old,signal.SIGTERM)
 print(json.dumps(rec))
