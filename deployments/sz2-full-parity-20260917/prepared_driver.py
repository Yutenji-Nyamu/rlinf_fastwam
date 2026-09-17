"""Execute a previously prepared explicit run; not called during deployment."""
import json,os,pathlib,runpy,signal,subprocess,sys
def main():
 runtime=pathlib.Path(sys.argv[1]).resolve();plan=json.loads((runtime/'plan.json').read_text())
 assert os.getuid()==20001 and runtime.is_relative_to(pathlib.Path('/data/chenyiteng/results').resolve())
 assert not (runtime/'driver-started.json').exists(),'This prepared run has already started.'
 repo=pathlib.Path(plan['repo']);head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();assert head==plan['head']
 for gpu in plan['gpus']:
  busy=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  assert not busy,'Requested GPU is occupied: '+str(gpu)
 os.environ.update(plan['environment']);sys.path[:0]=plan['environment']['PYTHONPATH'].split(':')
 from rlinf.scheduler import Cluster
 Cluster.NAMESPACE=plan['namespace']
 (runtime/'driver-started.json').write_text(json.dumps({'pid':os.getpid(),'uid':os.getuid(),'namespace':plan['namespace']}))
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 entry=plan['entry'];sys.argv=[entry,'--config-path',str(runtime),'--config-name','resolved','hydra.run.dir=.','hydra.output_subdir=null','hydra.job.chdir=false','hydra/job_logging=stdout'];os.chdir(repo)
 try:runpy.run_path(entry,run_name='__main__')
 finally:
  import ray
  if ray.is_initialized():
   job=ray.get_runtime_context().get_job_id()
   if hasattr(job,'hex'):job=job.hex()
   from ray.util.state import list_actors
   killed=[]
   try:
    for obj in list_actors(detail=True,filters=[('state','=','ALIVE')],limit=10000):
     row=obj.asdict() if hasattr(obj,'asdict') else dict(obj)
     if row.get('ray_namespace')!=plan['namespace'] or str(row.get('job_id'))!=str(job):continue
     pid=row.get('pid');status=pathlib.Path('/proc')/str(pid)/'status'
     if not status.exists():continue
     uidline=next(x for x in status.read_text().splitlines() if x.startswith('Uid:'))
     if int(uidline.split()[1])!=os.getuid():continue
     name=row.get('name')
     if name:ray.kill(ray.get_actor(name,namespace=plan['namespace']),no_restart=True);killed.append(name)
    (runtime/'owned-cleanup.json').write_text(json.dumps({'namespace':plan['namespace'],'job':str(job),'killed':killed}))
   finally:ray.shutdown()
if __name__=='__main__':main()
