"""Launch the single prepared owner once; its original finally returns RLT."""
import datetime,hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/click-bell-v2')
ready=json.loads((D/'prepared/ready.json').read_text());plan_path=Path(ready['plan'])
assert hashlib.sha256(plan_path.read_bytes()).hexdigest()==ready['sha256']
plan=json.loads(plan_path.read_text());assert not Path(plan['owner_dir']).exists()
assert not (D/'launch.json').exists()
argv=[plan['python'],'-u','-B',ready['entrypoint'],'--plan',str(plan_path),'owner']
with (D/'owner-console.log').open('x') as stream:
 p=subprocess.Popen(argv,cwd=plan['repo'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc')/str(p.pid)/'stat';v=stat.read_text();start=int(v[v.rfind(')')+2:].split()[19])
row=dict(time=datetime.datetime.now().astimezone().isoformat(),pid=p.pid,starttime=start,uid=os.getuid(),argv=argv,owner_dir=plan['owner_dir'],plan_sha256=ready['sha256'])
(D/'launch.json').write_text(json.dumps(row,indent=2)+'\n')
time.sleep(2);row['early_exit_code']=p.poll();print(json.dumps(row),flush=True)
