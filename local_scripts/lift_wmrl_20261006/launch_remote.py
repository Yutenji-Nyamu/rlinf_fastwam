"""Launch the prepared task owner exactly once; original finally returns RLT."""
import datetime,hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/lift-pot-v1')
ready=json.loads((D/'prepared/ready.json').read_text());p=Path(ready['plan'])
assert ready['cpu_validate_passed'] and hashlib.sha256(p.read_bytes()).hexdigest()==ready['plan_sha256']
plan=json.loads(p.read_text());assert not Path(plan['owner_dir']).exists()
assert plan['physical_gpus']==[4,5,6,7] and not (D/'launch.json').exists()
argv=[plan['python'],'-u','-B',ready['entrypoint'],'--plan',str(p),'owner']
with (D/'owner-console.log').open('x') as stream:
 child=subprocess.Popen(argv,cwd=plan['repo'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
proc=Path('/proc')/str(child.pid)
start=int((proc/'stat').read_text().split(') ',1)[1].split()[19])
row=dict(time=datetime.datetime.now().astimezone().isoformat(),pid=child.pid,starttime=start,uid=os.getuid(),argv=argv,owner_dir=plan['owner_dir'],plan_sha256=ready['plan_sha256'],cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
(D/'launch.json').write_text(json.dumps(row,indent=2)+'\n')
time.sleep(2);row['early_exit_code']=child.poll();print(json.dumps(row),flush=True)
