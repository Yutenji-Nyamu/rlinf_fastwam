"""Run one pinned RLinf recipe on a dedicated Ray instance on physical GPUs 4-7.

The outer resource owner is responsible for cleanup on success and failure.
Never calls ray stop or attaches to a discovered/shared Ray cluster.
"""
import argparse,atexit,json,os,socket,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--config',required=True)
p.add_argument('--repo',required=True)
p.add_argument('--log-dir',required=True)
p.add_argument('--port',type=int,required=True)
a=p.parse_args()
repo=Path(a.repo).resolve(); log=Path(a.log_dir).resolve(); log.mkdir(parents=True,exist_ok=True)
token=os.environ['WM_OWNER_TOKEN']
assert len(token)>=16
assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
namespace=os.environ['CLUSTER_NAMESPACE']
assert namespace.startswith('wan_goal_')
raytmp=Path(os.environ['WAN_GOAL_RAY_TMPDIR'])
assert str(raytmp).startswith('/data/chenyiteng/wr/') and len(str(raytmp))<45
assert not raytmp.exists(), 'Each Ray instance needs a fresh, unique temp directory'
raytmp.parent.mkdir(parents=True,exist_ok=True)
assert 1024<a.port<65535
with socket.socket() as s:s.bind(('127.0.0.1',a.port))
address=f'127.0.0.1:{a.port}'
assert os.environ['RAY_ADDRESS']==address
os.environ.update(RAY_USAGE_STATS_ENABLED='0',RAY_DEDUP_LOGS='0',
                  MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',ROBOT_PLATFORM='LIBERO',
                  EMBODIED_PATH=str(repo/'examples/embodiment'))
resource_directory=log/'resources'
resource_directory.mkdir(exist_ok=True)
os.environ['WAN_GOAL_RESOURCE_DIR']=str(resource_directory)
os.environ['PYTHONPATH']=str(repo)+os.pathsep+os.environ.get('PYTHONPATH','')
sys.path.insert(0,str(repo))
ray_cli=Path(sys.executable).parent/'ray'
cmd=[str(ray_cli),'start','--head','--node-ip-address=127.0.0.1',f'--port={a.port}',
     '--num-gpus=4','--num-cpus=64','--include-dashboard=false','--disable-usage-stats',
     '--object-store-memory=8589934592',f'--temp-dir={raytmp}']
receipt={'started':time.time(),'config':a.config,'repo':str(repo),'ray_address':address,
         'namespace':namespace,'ray_temp_dir':str(raytmp),'physical_gpus':[4,5,6,7],
         'ray_command':cmd,'driver_pid':os.getpid()}
receipt['resource_monitor']={'interval_seconds':10,'pss_interval_seconds':60,'boundary_dir':str(resource_directory)}
(log/'launch.json').write_text(json.dumps(receipt,indent=2))
monitor_log=(log/'resource-monitor.log').open('x')
monitor_process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('resource_monitor.py')),
    '--run',str(log),'--output',str(log/'resources.jsonl'),'--interval','10'],
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=monitor_log,stderr=subprocess.STDOUT)
def stop_resource_monitor():
    if monitor_process.poll() is None:
        monitor_process.terminate()
        try: monitor_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor_process.kill(); monitor_process.wait(timeout=5)
    monitor_log.close()
atexit.register(stop_resource_monitor)
subprocess.run(cmd,check=True,cwd=repo)

# Cluster's normal initialization connects to this explicit RAY_ADDRESS. Set the
# namespace before its managers are constructed; no change to learning code.
import ray
from rlinf.scheduler import Cluster
Cluster.NAMESPACE=namespace
orig_init=ray.init
def scoped_init(*args,**kwargs):
    requested=kwargs.get('address',args[0] if args else None)
    if requested in (None,'auto'):
        kwargs['address']=address
    elif requested!=address:
        raise RuntimeError(f'Unexpected Ray destination: {requested}')
    kwargs['namespace']=namespace
    runtime=kwargs.setdefault('runtime_env',{})
    runtime.setdefault('env_vars',{}).update(WM_OWNER_TOKEN=token,CLUSTER_NAMESPACE=namespace,RAY_ADDRESS=address,
                                          WAN_GOAL_RESOURCE_DIR=str(resource_directory))
    return orig_init(*args,**kwargs)
ray.init=scoped_init
from dataclasses import asdict
from rlinf.utils.placement import HybridComponentPlacement
original_placement_init=HybridComponentPlacement.__init__
def checked_placement_init(self,cfg,cluster):
    original_placement_init(self,cfg,cluster)
    placement_record={}
    for name in ('actor','env','rollout'):
        rows=self.get_strategy(name).get_placement(cluster,True)
        assert self.get_world_size(name)==4,(name,self.get_world_size(name))
        assert [r.visible_accelerators for r in rows]==[['4'],['5'],['6'],['7']],(name,rows)
        assert [r.cluster_node_rank for r in rows]==[0,0,0,0]
        placement_record[name]=[asdict(r) for r in rows]
    (log/'verified-placement.json').write_text(json.dumps(placement_record,indent=2,default=str))
    print('VERIFIED_PHYSICAL_PLACEMENT_4567',flush=True)
HybridComponentPlacement.__init__=checked_placement_init
sys.argv=[str(repo/'examples/embodiment/train_embodied_agent.py'),
          '--config-path',str(repo/'examples/embodiment/config'), '--config-name',a.config,
          f'runner.logger.log_path={log}']
import runpy
runpy.run_path(sys.argv[0],run_name='__main__')
