"""New attempt: correct native code-sync and log paths; retain completed pilot."""
import ast,base64,hashlib,json
from pathlib import Path
import make_deploy_rpc as first
HERE=Path(__file__).resolve().parent
def main():
 payload={}
 for name in first.FILES:
  path=HERE/('rm_owner_v2.py' if name=='rm_owner.py' else name)
  raw=path.read_bytes();ast.parse(raw,filename=str(path))
  payload[name]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 raw=(first.ROOT/'local_scripts/rynn_binary_20261005/run_native_eval.py').read_bytes()
 payload['run_native_eval.py']={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 remote=first.REMOTE.replace("D=S/'task-reward-v1'", "D=S/'task-reward-v2'")
 remote=remote.replace("C.mkdir(exist_ok=True,mode=0o700)", "C.mkdir(parents=True,exist_ok=True,mode=0o700)")
 start=remote.index("run(['-m','unittest'");end=remote.index('seeds=Path(',start)
 remote=remote[:start]+"# CPU tests and the bottle GPU pilot completed in v1; reuse their receipts.\n"+remote[end:]
 remote=remote.replace("'rm-lift128-v1'", "'rm-lift128-v2'").replace("'opendw_rm_lift128_v1'", "'opendw_rm_lift128_v2'")
 remote=remote.replace("weights=D/'assets/resnet18-f37072fd.pth'", "weights=S/'task-reward-v1/assets/resnet18-f37072fd.pth'")
 remote=remote.replace("private=S/'rlinf-rynn-binary-v1'", "private=S/'rlinf-rynn-binary-v1'\nbase['RLINF_CODE_WORKING_DIR']=str(private)\nfrag['RLINF_CODE_WORKING_DIR']=str(private)")
 lines=remote.splitlines();lines=[line for line in lines if not line.startswith(" command('pilot'")];remote='\n'.join(lines)+'\n'
 remote=remote.replace("native/'run/worker-metrics'", "native/'run/worker_logs'")
 assert 'command(\'pilot\'' not in remote and 'opendw_rm_lift128_v1' not in remote
 source='PAYLOAD='+repr(payload)+'\n'+remote;ast.parse(source)
 (HERE/'stage_v2_remote.py').write_text(source,encoding='utf-8')
 print(json.dumps({'files':list(payload),'attempt':'task-reward-v2','native_code_sync':'rlinf-rynn-binary-v1','pilot_reused':True}))
if __name__=='__main__':main()
