"""One experiment; on exit clean only its exact Ray job and namespace."""
import argparse,os,resource,runpy,signal,sys,time
from pathlib import Path
from common import read,save,proc,actors
p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--plan',required=True);a=p.parse_args();q=read(a.request);plan=read(a.plan);rt=Path(q['runtime'])
save(rt/'driver-identity.json',dict(proc(os.getpid()),namespace=q['namespace'],time=time.time()))
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(max(soft,min(4096,hard)),hard))
signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
from rlinf.scheduler import Cluster
Cluster.NAMESPACE=q['namespace'];entry=str(Path(q['repo'])/'examples/embodiment/train_embodied_agent.py')
sys.argv=[entry,'--config-path',str(rt),'--config-name','resolved','hydra.run.dir=.','hydra.output_subdir=null','hydra.job.chdir=false','hydra/job_logging=stdout']
rc=0;error=None
try:runpy.run_path(entry,run_name='__main__')
except SystemExit as e:rc=e.code if isinstance(e.code,int) else 1
except BaseException:
 import traceback;traceback.print_exc();rc=1
finally:
 import ray
 signal.signal(signal.SIGUSR1,signal.SIG_IGN)
 try:
  if ray.is_initialized():
   job=ray.get_runtime_context().get_job_id();job=job.hex() if hasattr(job,'hex') else str(job);selected=actors(plan,q['namespace'])
   for row in selected:
    assert row['job_id']==job and row.get('name') and row.get('actor_id');who=proc(row.get('pid',0));assert not who or who['uid']==os.getuid()
   save(rt/'cleanup-targets.json',dict(time=time.time(),job_id=job,actors=selected))
   for row in selected:
    try:handle=ray.get_actor(row['name'],namespace=q['namespace'])
    except ValueError:continue
    assert handle._actor_id.hex()==row['actor_id'];ray.kill(handle,no_restart=True)
 except BaseException as e:error=repr(e);rc=1
 finally:ray.shutdown();save(rt/'finished.json',dict(time=time.time(),exit_code=rc,cleanup_error=error))
sys.exit(rc)
