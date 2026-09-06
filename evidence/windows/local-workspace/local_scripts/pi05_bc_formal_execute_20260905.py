"""Explicit authorized formal100 packet, unique launch and read-only startup check."""
import argparse
import getpass
import json
import os
from pathlib import Path
import remote_exec_autodl as ssh

ROOT='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc'
RUN='/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1'
HEAD='912bc6907d39a0eec1eb98a6d4c9358e69791924'
DOC=Path('docs/rlinf-robotwin-pi0-online-bc/evidence')
PFX='PI05_BC_FORMAL_'

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','launch','status','publish'));args=p.parse_args()
 os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
 client=ssh.connect(argparse.Namespace(host='120.241.223.9',port=22,user='chenyiteng',timeout=20,host_key_sha256='qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'))
 def run(command):
  _,out,err=client.exec_command(command)
  data=out.read().decode(errors='replace'); error=err.read().decode(errors='replace');rc=out.channel.recv_exit_status()
  if error:print(error[-3000:],flush=True)
  assert rc==0,(rc,data[-4000:],error[-3000:])
  return data
 try:
  print(run('id; date -Is'),flush=True)
  if args.mode=='prepare':
   raw=run(Path('local_scripts/remote_commands/sz_pi05_bc_formal_prepare_20260905.sh').read_text(encoding='utf-8'))
   # Framework banners can precede the single JSON object.
   result=json.loads(raw.splitlines()[-1]); assert result['passed']
   (DOC/(PFX+'PREFLIGHT_20260905.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
   (DOC/(PFX+'RESOLVED_20260905.yaml')).write_text(result.pop('resolved_yaml'),encoding='utf-8')
   print(json.dumps(result,indent=2),flush=True)
  elif args.mode=='launch':
   proof=json.loads((DOC/(PFX+'PREFLIGHT_20260905.json')).read_text(encoding='utf-8'))
   assert proof['passed'] and proof['run']==RUN and proof['head']==HEAD
   run(f'''set -eu
test "$(git -C {ROOT} rev-parse HEAD)" = {HEAD}
test -z "$(git -C {ROOT} status --porcelain)"
test ! -e {RUN}
test -z "$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader,nounits)"
''')
   with client.open_sftp() as sftp:
    sftp.mkdir(RUN);sftp.mkdir(RUN+'/runtime')
    uploads={'wrapper.sh':Path('local_scripts/pi05_bc_formal_wrapper_20260905.sh'),'resource_observer.py':Path('local_scripts/bc_gpu6_resource_observer_20260905.py'),'resolved.yaml':DOC/(PFX+'RESOLVED_20260905.yaml'),'contract.md':DOC/'GPU6_PI05_BC_FORMAL_CONTRACT_20260905.md'}
    for name,source in uploads.items():
     with sftp.open(RUN+'/runtime/'+name,'wb') as target:target.write(source.read_bytes())
   # Exactly one launch. Never automatically replay this command after uncertainty.
   data=run(f'''set -eu
bash -n {RUN}/runtime/wrapper.sh
test ! -e {RUN}/wrapper.pid
nohup setsid bash {RUN}/runtime/wrapper.sh > {RUN}/wrapper.log 2>&1 < /dev/null & p=$!
printf '%s\n' "$p" > {RUN}/wrapper.pid
nohup /usr/bin/python3 {RUN}/runtime/resource_observer.py "$p" {RUN} > {RUN}/observer.log 2>&1 < /dev/null & o=$!
printf '%s\n' "$o" > {RUN}/observer.pid
printf 'WRAPPER_PID=%s OBSERVER_PID=%s\n' "$p" "$o"
''')
   (DOC/(PFX+'LAUNCH_20260905.txt')).write_text(data,encoding='utf-8');print(data,flush=True)
  elif args.mode=='status':
   source=Path('local_scripts/remote_commands/sz_pi05_bc_status_20260905.sh').read_text(encoding='utf-8')
   source=source.replace('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1',RUN)
   raw=run(source);result=json.loads(raw)
   (DOC/(PFX+'STARTUP_20260905.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print(raw,flush=True)
  elif args.mode=='publish':
   run(f'test "$(git -C {ROOT} rev-parse HEAD)" = {HEAD} && test -z "$(git -C {ROOT} status --porcelain)"')
   names=[PFX+'PREFLIGHT_20260905.json',PFX+'RESOLVED_20260905.yaml',PFX+'STARTUP_20260905.json',PFX+'BINDING_20260905.json',PFX+'LAUNCH_20260905.txt','GPU6_PI05_BC_FORMAL_CONTRACT_20260905.md','GPU6_PI05_BC_FORMAL_LAUNCH_LEDGER_20260905.md']
   uploads={n:DOC/n for n in names}
   uploads['pi05_bc_formal_wrapper_20260905.sh']=Path('local_scripts/pi05_bc_formal_wrapper_20260905.sh')
   uploads['PARAMETER_SOURCES.md']=Path('docs/rlinf-robotwin-pi0-online-bc/02_PI05_ONLINE_BC_PLAN.md')
   target=ROOT+'/docs/evidence/pi05-online-bc-formal100-20260905'
   with client.open_sftp() as sftp:
    sftp.mkdir(target)
    for n,source in uploads.items():
     raw='\n'.join(x.rstrip() for x in source.read_text(encoding='utf-8').splitlines())+'\n'
     assert len(raw.encode())<1000000
     with sftp.open(target+'/'+n,'wb') as f:f.write(raw.encode())
   data=run(f'''set -eu
cd {ROOT}
git add -f -- {' '.join('docs/evidence/pi05-online-bc-formal100-20260905/'+n for n in uploads)}
git diff --cached --check
git commit -m 'Record authorized pi05 BC GPU6 formal100 launch and parameter sources'
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 60s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/codex/sz-pi05-online-bc
git rev-parse HEAD
git status --porcelain
''')
   (DOC/(PFX+'PUSH_20260905.txt')).write_text(data,encoding='utf-8');print(data,flush=True)
 finally:
  client.close();os.environ.pop('SEETA_SSH_PASSWORD',None)

if __name__=='__main__':main()
