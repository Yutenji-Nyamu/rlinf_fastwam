"""Launch the prepared B16 formal owner once; no smoke or retry supervisor."""
import datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
F=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/lift-two-gpu-b16-lean-20261007-v1')
read=lambda p:json.loads(Path(p).read_text())
ready=read(F/'ready.json');plan=read(ready['plan'])
assert ready['skip_smoke'] and ready['config_equal_except_output_names']
assert hashlib.sha256(Path(ready['plan']).read_bytes()).hexdigest()==ready['plan_sha256']
assert not Path(plan['owner_dir']).exists() and not (F/'launch.json').exists()
# The owner validates the frozen plan once on entry; do not repeat the same
# full import/validation in this short-lived launcher.
argv=[plan['python'],'-u','-B',ready['entrypoint'],'--plan',ready['plan'],'owner']
with (F/'owner-console.log').open('x') as stream:
 child=subprocess.Popen(argv,cwd=plan['repo'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1'),
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
p=Path('/proc')/str(child.pid);st=(p/'stat').read_text().rsplit(')',1)[1].split();assert p.stat().st_uid==20001
value={'time':datetime.datetime.now().astimezone().isoformat(),'pid':child.pid,'start':int(st[19]),'uid':20001,
       'argv':argv,'plan_sha256':ready['plan_sha256'],'physical_gpus':[4,5],'wm_batch':16,'skip_smoke':True,'max_steps':200}
(F/'launch.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
