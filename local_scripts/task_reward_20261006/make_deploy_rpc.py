"""Serialize exact reviewed RM files for CPU-only remote staging; no SSH here."""
import ast,base64,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FILES=['rm_owner.py','rm_train.py','rm_inference.py','prepare_dataset.py','test_task_reward.py','prepare_native_capture.py','validate_capture.py']
def main():
 payload={}
 for name in FILES:
  raw=(HERE/name).read_bytes();ast.parse(raw,filename=name)
  payload[name]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 name='run_native_eval.py';raw=(ROOT/'local_scripts/rynn_binary_20261005'/name).read_bytes()
 payload[name]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 remote='PAYLOAD='+repr(payload)+'\n'+REMOTE
 (HERE/'stage_remote.py').write_text(remote,encoding='utf-8')
 print(json.dumps({'files':list(payload),'bytes':len(remote)}))
REMOTE=r'''
import ast,base64,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'task-reward-v1';C=D/'code'
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
C.mkdir(exist_ok=True,mode=0o700)
for name,row in PAYLOAD.items():
 raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];ast.parse(raw)
 path=C/name
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as stream:stream.write(raw)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
def run(argv,timeout=120):
 p=subprocess.run([PY,'-B',*map(str,argv)],cwd=C,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
 print(json.dumps({'argv':list(map(str,argv)),'returncode':p.returncode,'output':p.stdout[-16000:]}),flush=True)
 assert p.returncode==0
run(['-m','unittest','-v','test_task_reward.py'])
data=D/'data/bottle-pilot'
if not (data/'manifest.json').exists():
 run([C/'prepare_dataset.py','--capture-dir',S/'runs/click-bell-v2/native_rynn32/capture','--task','adjust_bottle','--task-config','demo_clean','--output-dir',data])
seeds=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-rlt-step-timeout-sz3-20261003/rlinf/envs/robotwin/seeds')
run([C/'prepare_native_capture.py','--config',S/'click-bell-v2/prepared/formal.yaml','--task','lift_pot','--seed-file',seeds/'lift_pot_train_rlt_20261002.json','--exclude-seed-file',seeds/'lift_pot_eval_rlt_20261002.json','--output',D/'native-lift128','--name','rm-lift128-v1'])
old=json.loads((S/'runs/click-bell-v2/owner-plan.json').read_text())
base=json.loads(Path(old['environment_file']).read_text())
for k in list(base):
 if k in ['CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','PYTHONPATH','LD_PRELOAD','OPENDW_SMOKE_OWNER_TOKEN','OPENDW_SMOKE_OWNER_PHASE'] or k.startswith(('RLINF_OPENDW_','__GL_','RYNN_BINARY_')):base.pop(k)
base.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
scope=S/'rynn-numeric-v1/rlt-return-repair-v2/scope'
frag=json.loads((scope/'environment-fragment.json').read_text())
private=S/'rlinf-rynn-binary-v1'
parts=frag['PYTHONPATH'].split(':');parts=[str(private) if p==str(S/'rlinf-multigpu-v1') else p for p in parts]
frag['PYTHONPATH']=':'.join(parts)
prepared=D/'prepared';prepared.mkdir(exist_ok=True,mode=0o700)
def exact_json(path,value):
 raw=(json.dumps(value,indent=2)+'\n').encode()
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as f:f.write(raw)
  path.chmod(0o600)
fragment=prepared/'native-environment.json';exact_json(fragment,frag)
O=D/'run';native=D/'native-lift128';weights=D/'assets/resnet18-f37072fd.pth'
def command(key,kind,argv,timeout,result=None,checks=None,**kw):
 value=dict(key=key,kind=kind,argv=[PY,'-u','-B',*map(str,argv)],cwd=str(C),timeout_seconds=timeout,**kw)
 if result:value['result_file']=str(result)
 if checks:value['result_checks']=checks
 return value
commands=[
 command('pilot','gpu',[C/'rm_train.py','--dataset-dir',data,'--output-dir',D/'pilot','--pretrained-path',weights,'--epochs','20','--patience','8','--profile','--auto-micro'],1800,D/'pilot/report.json',{'task_name':'adjust_bottle'}),
 command('native_lift128','native',[C/'run_native_eval.py','--config',native/'native_eval.json','--receipt-dir',O/'native_lift128','--private-repo',private,'--environment-fragment',fragment,'--capture-dir',native/'capture','--capture-mode','reward_native','--namespace','opendw_rm_lift128_v1','--ray-address','127.0.0.1:26379'],10800,namespace='opendw_rm_lift128_v1',config=str(native/'native_eval.json'),environment_fragment=str(fragment)),
 command('validate_capture','cpu',[C/'validate_capture.py','--capture-dir',native/'capture','--plan',native/'capture_plan.json','--worker-log-dir',native/'run/worker-metrics','--output',D/'capture-validation.json'],600,D/'capture-validation.json',{'passed':True}),
 command('prepare_lift_data','cpu',[C/'prepare_dataset.py','--capture-dir',native/'capture','--task','lift_pot','--task-config','demo_clean','--output-dir',D/'data/lift-pot'],600,D/'data/lift-pot/manifest.json',{'task_name':'lift_pot'}),
 command('train_lift','gpu',[C/'rm_train.py','--dataset-dir',D/'data/lift-pot','--output-dir',D/'lift-pot','--pretrained-path',weights,'--profile','--auto-micro'],3600,D/'lift-pot/report.json',{'task_name':'lift_pot'})]
files=[*C.glob('*.py'),fragment,native/'native_eval.json',native/'native_seeds.json',native/'capture_plan.json',weights,
 private/'rlinf/envs/robotwin/robotwin_env.py',private/'rlinf/envs/robotwin/native_binary_recorder.py']
source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
spec=dict(commands=commands,environment=base,source_sha256=source)
exact_json(prepared/'spec.json',spec)
print(json.dumps({'staged':True,'spec':str(prepared/'spec.json'),'source_files':len(source),'commands':[x['key'] for x in commands]}))
'''
if __name__=='__main__':main()
