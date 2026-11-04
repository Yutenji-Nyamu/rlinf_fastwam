"""Accepted Clean GRPO -> fixed final-policy pool -> six-point replay.

No training algorithm changes. The following FQL queue owns the eventual
handoff; this owner always releases on a bounded failure.
"""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,'/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops')
from common import read,save,proc,same,exact_signal,gpus,admission_card

def main(directory):
 d=Path(directory);p=read(d/'plan.json');me=proc(os.getpid())
 save(d/'identity.json',me);save(d/'status.json',{'phase':'WAIT_PAPER_REPLAY','time':time.time()})
 while not (d/'registered.json').exists():time.sleep(.1)
 assert read(d/'registered.json')['identity']['pid']==me['pid']
 while same(p['predecessor_identity']) or any(admission_card(gpus()[g],p.get('gpu_admission',{}).get(str(g),{}))['processes'] for g in p.get('gpus',[4,5])):time.sleep(.25)
 result={};child=None
 try:
  env=dict(os.environ,**read(d/'training-environment.json'))
  for k in ['CUDA_VISIBLE_DEVICES','DOJO_GPU_SCOPE']:env.pop(k,None)
  env.update(GRPO_ENTROPY_DEPLOYMENT=str(d),PYTHONUNBUFFERED='1')
  command=[p['python'],'-u','-B',str(d/'tools/grpo_entropy_driver.py'),'--config-path',str(d/'config'),'--config-name','grpo_clean']
  with (d/'driver.log').open('x') as log:
   child=subprocess.Popen(command,cwd=p['training_source'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   who=proc(child.pid);save(d/'status.json',{'phase':'GRPO_TRAINING','time':time.time(),'child':who,'rounds':200,'checkpoints':[0,40,80,120,160,200]})
   try:rc=child.wait(timeout=345600)
   except subprocess.TimeoutExpired:
    exact_signal(who,signal.SIGTERM)
    child.wait(timeout=180)
    raise TimeoutError('Accepted four-day training deadline reached; release to FQL')
  if rc:raise RuntimeError(f'Clean GRPO exited {rc}; see driver.log')
  checkpoint=Path(p['checkpoint_root'])/'global_step_200/actor/model_state_dict/full_weights.pt'
  if not checkpoint.is_file():raise RuntimeError('Final CP200 missing; do not label partial training complete')
  save(d/'status.json',{'phase':'FINAL_POOL_AND_SIX_POINT_REPLAY','time':time.time()})
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
  with (d/'six-point.log').open('x') as log:
   child=subprocess.Popen([p['python'],'-u','-B',str(d/'tools/six_point_after.py'),str(d)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   who=proc(child.pid)
   try:rc=child.wait(timeout=14400)
   except subprocess.TimeoutExpired:
    exact_signal(who,signal.SIGTERM);child.wait(timeout=60)
    raise TimeoutError('Post-training collection exceeded four hours')
  if rc:raise RuntimeError(f'Six-point collection exited {rc}; keep its partial raw data')
  result={'phase':'COMPLETE','time':time.time(),'output':p['output']}
 except BaseException:
  result={'phase':'FAILED_ADVANCE_FQL','time':time.time(),'error':traceback.format_exc()}
 finally:
  save(d/'status.json',result);save(d/'released.json',result)

if __name__=='__main__':main(sys.argv[1])
