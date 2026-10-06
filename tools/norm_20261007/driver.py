"""Run one frozen request and clean only its verified Ray job."""
import argparse,importlib.util,json,os,resource,runpy,signal,sys,time,traceback
from pathlib import Path
def load(p):
    s=importlib.util.spec_from_file_location('norm_driver_helper',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--control',required=True);p.add_argument('--request',required=True);a=p.parse_args()
    C=Path(a.control);plan=json.loads((C/'plan.json').read_text());req=json.loads(Path(a.request).read_text());rt=Path(req['runtime'])
    H=load(Path(plan['cycle'])/'rlt_checkpoint_lifecycle.py')
    me=H.proc(os.getpid());H.save(rt/'driver-identity.json',dict(me,namespace=req['namespace'],request=a.request,time=H.now()))
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(max(soft,min(4096,hard)),hard))
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    from rlinf.scheduler import Cluster
    Cluster.NAMESPACE=req['namespace']
    entry=str(Path(req['repo'])/'examples/embodiment/train_embodied_agent.py')
    sys.argv=[entry,'--config-path',str(rt),'--config-name','resolved','hydra.run.dir=.','hydra.output_subdir=null','hydra.job.chdir=false','hydra/job_logging=stdout']
    rc=0;error=None
    try:runpy.run_path(entry,run_name='__main__')
    except SystemExit as e:rc=e.code if isinstance(e.code,int) else 1;raise
    except BaseException:rc=1;raise
    finally:
        import ray
        signal.signal(signal.SIGUSR1,signal.SIG_IGN)
        try:
            if ray.is_initialized():
                job=ray.get_runtime_context().get_job_id();job=job.hex() if hasattr(job,'hex') else str(job)
                selected=H.active(H.actors(plan),req['namespace']);H.validate_actor_rows(selected,req['namespace'],{job})
                H.save(rt/'cleanup-targets.json',{'time':H.now(),'job_id':job,'actors':selected})
                for actor in selected:
                    try:handle=ray.get_actor(actor['name'],namespace=req['namespace'])
                    except ValueError:continue
                    assert handle._actor_id.hex()==actor['actor_id'];ray.kill(handle,no_restart=True)
        except BaseException as e:error=repr(e);rc=1
        finally:
            ray.shutdown();H.save(rt/'finished.json',{'time':H.now(),'exit_code':rc,'cleanup_error':error})
        if error:raise RuntimeError(error)
if __name__=='__main__':main()
