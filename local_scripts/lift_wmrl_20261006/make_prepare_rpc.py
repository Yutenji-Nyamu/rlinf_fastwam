"""Package exact new task source and prepare assets on SZ3 without GPU work."""
import ast,base64,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FILES=['rm_adapter.py','patch_lift.py','test_rm_adapter.py','reset_prepare.py','native_seed_prepare.py','recipe_prepare.py','lifecycle_prepare.py']
REMOTE=r'''
import hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'lift-pot-v1';C=D/'code'
P='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
W=str(S/'venv/bin/python')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
C.mkdir(parents=True,exist_ok=True,mode=0o700)
for name,row in PAYLOAD.items():
 raw=__import__('base64').b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 target=C/name
 if target.exists():assert target.read_bytes()==raw,'Source changed: '+str(target)
 else:target.write_bytes(raw);target.chmod(0o500)
def run(argv):
 result=subprocess.run(argv,cwd=C,env=dict(os.environ,PYTHONPATH=str(C),OMP_NUM_THREADS='1'),text=True,capture_output=True,timeout=180)
 print(json.dumps({'argv':argv,'rc':result.returncode,'stdout':result.stdout[-12000:],'stderr':result.stderr[-5000:]}),flush=True)
 assert result.returncode==0
run([W,'-B','-m','unittest','-v','test_rm_adapter'])
assets=D/'assets';assets.mkdir(exist_ok=True)
reset=assets/'lift_pot_clean50_reset.npz'
if not reset.exists():run([P,'-B',str(C/'reset_prepare.py'),'--output',str(reset)])
seeds=assets/'native_seeds32.json'
if not seeds.exists():run([P,'-B',str(C/'native_seed_prepare.py'),'--source','/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-rlt-step-timeout-sz3-20261003/rlinf/envs/robotwin/seeds/lift_pot_train_rlt_20261002.json','--rm-capture-seeds',str(S/'task-reward-v2/native-lift128/native_seeds.json'),'--output',str(seeds)])
base=S/'click-bell-v2/prepared/formal.yaml'
if not (D/'prepared').exists():run([P,'-B',str(C/'recipe_prepare.py'),'--base-config',str(base),'--base-config-sha256','1de2f704c743d623dbad60d8f78d8eeeec19ffb6e0cbaa683044c5da96e6b158','--reward-report',str(S/'task-reward-v2/lift-pot/report.json'),'--reward-checkpoint',str(S/'task-reward-v2/lift-pot/best.pt'),'--name','lift-pot-v1','--owner-dir',str(S/'runs/lift-pot-v1'),'--initial-state',str(reset),'--native-seeds',str(seeds),'--sft-path','/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab','--service-urls','http://127.0.0.1:18976','http://127.0.0.1:18977','--output',str(D/'prepared')])
if not (D/'prepared-cycles').exists():run([P,'-B',str(C/'lifecycle_prepare.py'),'--prepared-dir',str(D/'prepared-cycles')])
print(json.dumps({'cpu_prepared':True,'launched':False,'new_root':str(D),'files':{str(C/n):sha(C/n) for n in PAYLOAD}}))
'''
def main():
 payload={}
 paths={n:HERE/n for n in FILES}
 paths['rm_inference.py']=ROOT/'local_scripts/task_reward_20261006/rm_inference.py'
 for name,path in paths.items():
  raw=path.read_bytes();ast.parse(raw);payload[name]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 source='PAYLOAD='+repr(payload)+'\n'+REMOTE;ast.parse(source)
 (HERE/'prepare_remote.py').write_text(source,encoding='utf-8')
 print(json.dumps({'files':list(payload),'gpu_work':False}))
if __name__=='__main__':main()
