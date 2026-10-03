"""Build the monitored driver from the exact driver used by the stopped r6 run."""
import argparse,hashlib
from pathlib import Path

def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new,1)

def build(source):
    data=source.read_bytes()
    assert hashlib.sha256(data).hexdigest()=='6b36a2b6d0e87c4f481a19530758a4716c07fb7239f10f3dc8d03d4eaab18242'
    text=data.decode()
    text=replace_once(text,'import argparse,json,os,socket,subprocess,sys,time\n','import argparse,atexit,json,os,socket,subprocess,sys,time\n')
    text=replace_once(text,"os.environ['PYTHONPATH']=str(repo)+os.pathsep+os.environ.get('PYTHONPATH','')\n",
        "resource_directory=log/'resources'\n"
        "resource_directory.mkdir(exist_ok=True)\n"
        "os.environ['WAN_GOAL_RESOURCE_DIR']=str(resource_directory)\n"
        "os.environ['PYTHONPATH']=str(repo)+os.pathsep+os.environ.get('PYTHONPATH','')\n")
    text=replace_once(text,"(log/'launch.json').write_text(json.dumps(receipt,indent=2))\nsubprocess.run(cmd,check=True,cwd=repo)\n",
        "receipt['resource_monitor']={'interval_seconds':10,'pss_interval_seconds':60,'boundary_dir':str(resource_directory)}\n"
        "(log/'launch.json').write_text(json.dumps(receipt,indent=2))\n"
        "monitor_log=(log/'resource-monitor.log').open('x')\n"
        "monitor_process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('resource_monitor.py')),\n"
        "    '--run',str(log),'--output',str(log/'resources.jsonl'),'--interval','10'],\n"
        "    env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=monitor_log,stderr=subprocess.STDOUT)\n"
        "def stop_resource_monitor():\n"
        "    if monitor_process.poll() is None:\n"
        "        monitor_process.terminate()\n"
        "        try: monitor_process.wait(timeout=10)\n"
        "        except subprocess.TimeoutExpired:\n"
        "            monitor_process.kill(); monitor_process.wait(timeout=5)\n"
        "    monitor_log.close()\n"
        "atexit.register(stop_resource_monitor)\n"
        "subprocess.run(cmd,check=True,cwd=repo)\n")
    text=replace_once(text,"runtime.setdefault('env_vars',{}).update(WM_OWNER_TOKEN=token,CLUSTER_NAMESPACE=namespace,RAY_ADDRESS=address)\n",
        "runtime.setdefault('env_vars',{}).update(WM_OWNER_TOKEN=token,CLUSTER_NAMESPACE=namespace,RAY_ADDRESS=address,\n"
        "                                          WAN_GOAL_RESOURCE_DIR=str(resource_directory))\n")
    compile(text,'private_ray_driver_monitored.py','exec')
    return text.encode()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=build(a.source)
    with a.output.open('xb') as f:f.write(data)
    print(hashlib.sha256(data).hexdigest())
