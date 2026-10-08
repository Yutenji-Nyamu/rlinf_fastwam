"""One CPU owner per host: independent priority -> original RLT per card."""
import argparse,fcntl,importlib.util,os,signal,subprocess,sys,time,traceback,zipfile
from pathlib import Path
from common import read,save,proc,same,exact_signal,actors,gpus,releasable

def launch(plan,request):
 q=read(request);rt=Path(q['runtime']);assert not (rt/'started.json').exists(), 'Existing dispatch requires inspection'
 env=dict(os.environ,**read(rt/'environment.json'),CLUSTER_NAMESPACE=q['namespace'])
 for k in ['CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES']:env.pop(k,None)
 with (rt/'driver.log').open('x') as log:
  child=subprocess.Popen([plan['python'],'-u','-B',str(Path(__file__).with_name('driver.py')),'--plan',plan['plan_path'],'--request',request],cwd=q['repo'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ident=proc(child.pid);save(rt/'started.json',dict(time=time.time(),identity=ident,request=request));return ident,child

def retain_latest(q):
 if q.get('kind')!='bc':return
 run=Path(q['run']);root=run/run.name/'checkpoints';complete=[]
 for cp in root.glob('global_step_*'):
  required=['actor/local_shard_checkpoint/checkpoint_rank_0.pt','actor/online_bc/rank_0/learner.pt','actor/online_bc/rank_0/success_replay.pt','actor/online_bc/rank_0/signal_tau.pt']
  if all((cp/f).is_file() and zipfile.is_zipfile(cp/f) for f in required):complete.append(cp)
 for cp in sorted(complete,key=lambda p:int(p.name.split('_')[-1]))[:-1]:
  for relative in ['actor/local_shard_checkpoint/checkpoint_rank_0.pt','actor/model_state_dict/full_weights.pt','actor/online_bc/rank_0/success_replay.pt']:
   f=cp/relative
   if not f.is_file():continue
   st=f.lstat();assert not f.is_symlink() and f.resolve().is_relative_to(run.resolve()) and st.st_uid==os.getuid()
   if st.st_size>1024**3:
    with (run/'runtime/retention.jsonl').open('a') as log:log.write(__import__('json').dumps(dict(time=time.time(),path=str(f),bytes=st.st_size,reason='keep latest complete BC checkpoint'))+'\n')
    f.unlink()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);C=Path(a.plan).parent
 assert os.getuid()==p['uid'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==p['boot_id']
 lock=(C/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 me=proc(os.getpid());save(C/'owner-identity.json',me);state=read(C/'status.json') if (C/'status.json').exists() else {'slots':{g:{'phase':'PENDING_PRIORITY'} for g in p['slots']}}
 for g,slot in p['slots'].items():state['slots'].setdefault(g,{'phase':'WAIT_EXTERNAL' if 'external_release' in slot else 'PENDING_PRIORITY'})
 stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));children={}
 while not stop[0]:
  try:
   snapshot=gpus();state.pop('gpu_query_error',None)
  except Exception:
   state.update(time=time.time(),owner=me,gpu_query_error=traceback.format_exc());save(C/'status.json',state)
   time.sleep(15);continue
  for g,slot in p['slots'].items():
   row=state['slots'][g]
   try:
    card=snapshot[int(g)]
    assert card['uuid']==slot['gpu_uuid']
    if row['phase']=='WAIT_EXTERNAL':
     if Path(slot['external_release']).is_file() and not same(slot['external_identity']) and not card['processes']:row['phase']='WAITING_RLT'
    elif row['phase']=='PENDING_PRIORITY':
     if card['processes']:row['waiting']='GPU_OCCUPIED';continue
     row['identity'],children[g]=launch(p,slot['priority']);row.update(phase='PRIORITY_RUNNING',request=slot['priority'],started_at=time.time());row.pop('waiting',None)
    elif row['phase'] in ['PRIORITY_RUNNING','RLT_RUNNING']:
     q=read(row['request'])
     if same(row['identity']):
      for index,record in snapshot.items():
       for process in record['processes']:
        try:pe=dict(t.split('=',1) for t in (Path('/proc')/str(process['pid'])/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in t)
        except OSError:continue
        if pe.get('CLUSTER_NAMESPACE')==q['namespace'] and index!=int(g):
         exact_signal(row['identity'],signal.SIGTERM);raise RuntimeError('Own context escaped assigned GPU')
      if row['phase']=='PRIORITY_RUNNING' and time.time()-row.get('retention_checked',0)>=300:
       try:retain_latest(q);row.pop('retention_error',None)
       except Exception:row['retention_error']=traceback.format_exc()
       row['retention_checked']=time.time()
      continue
     if g in children:children[g].poll()
     if not releasable(row['identity'],actors(p,q['namespace']),card):row['waiting']='EXIT_CLEANUP';continue
     save(C/('released-g'+g+'-'+row['phase']+'.json'),dict(time=time.time(),request=row['request'],identity=row['identity'],namespace_clear=True,gpu_clear=True))
     if row['phase']=='PRIORITY_RUNNING':row['phase']='WAITING_RLT'
     else:row['phase']='RLT_EXITED'
    elif row['phase']=='WAITING_RLT':
     if card['processes']:row['waiting']='GPU_OCCUPIED';continue
     if not (C/'fallback-enabled.json').exists():row['waiting']='PREPARATION_HOLD';continue
     row['identity'],children[g]=launch(p,slot['fallback']);row.update(phase='RLT_RUNNING',request=slot['fallback'],started_at=time.time());row.pop('waiting',None)
   except Exception:row['error']=traceback.format_exc()
  state.update(time=time.time(),owner=me);save(C/'status.json',state)
  time.sleep(15)
 save(C/'owner-stopped.json',dict(time=time.time(),owner=me,children_retained=True))
if __name__=='__main__':main()
