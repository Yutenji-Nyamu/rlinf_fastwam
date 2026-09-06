"""Source-locked pi05 BC DVAC port and user-authorized direct GPU7 formal run.

No smoke mode, automatic job replay, dependency change or shared-service control.
"""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path

import remote_exec_autodl as ssh
from bc_dvac_execute_20260905 import FILES

BASE='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc'
ROOT=BASE+'-dvac'
HEAD='6a93605d91dbfdc321c5602108ccca5e2f044001'
BRANCH='codex/sz-pi05-online-bc-dvac'
OLD=BASE.replace('pi05-online-bc','pi0-online-bc-dvac')
PACKET='/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-pi05-dvac-20260905'
RUN='/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1'
LOCAL=Path('worktrees/pi05-online-bc-dvac')
DOC=Path('docs/rlinf-robotwin-pi0-online-bc/evidence')
PFX='PI05_BC_DVAC_'
EXTRA=['examples/embodiment/config/bc_dvac/bounded_half.yaml','tests/unit_tests/test_pi05_online_bc_dvac.py']

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','deploy','test','fetch','publish','launch','status','publish-startup','finish-startup'));a=p.parse_args()
 os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
 client=ssh.connect(argparse.Namespace(host='120.241.223.9',port=22,user='chenyiteng',timeout=20,host_key_sha256='qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'))
 def run(command, output=None):
  _,out,err=client.exec_command(command)
  raw=out.read().decode(errors='replace'); error=err.read().decode(errors='replace');rc=out.channel.recv_exit_status()
  if output:
   (DOC/output).write_text(raw if output.endswith('.json') else raw+'\n'+error,encoding='utf-8')
   if error:(DOC/(output+'.stderr.txt')).write_text(error,encoding='utf-8')
  print(raw[-8000:] if len(raw)<12000 else f'Captured {len(raw)} bytes; see {output}',flush=True)
  if error:print(error[-3000:],flush=True)
  assert rc==0,('Command failed; never automatically replay side effects',rc)
  return raw
 try:
  run('id; date -Is')
  if a.mode=='prepare':
   assert not LOCAL.exists()
   proof=json.loads((DOC/(PFX+'PREFLIGHT_20260905.json')).read_text(encoding='utf-8'))
   assert proof['passed'] and all(x['matches_old_dvac_parent'] for x in proof['shared_files'].values())
   run(f'''set -eu
test "$(git -C {BASE} rev-parse HEAD)" = {HEAD}
test -z "$(git -C {BASE} status --porcelain)"
test ! -e {ROOT}
test ! -e {PACKET}
test ! -e {RUN}
test -z "$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader,nounits)"
git -C {BASE} worktree add -b {BRANCH} {ROOT} {HEAD}
mkdir {PACKET}
git -C {OLD} diff 736b1416^ 736b1416 -- {' '.join(FILES)} > {PACKET}/original-dvac.patch
git -C {ROOT} apply --check {PACKET}/original-dvac.patch
git -C {ROOT} apply {PACKET}/original-dvac.patch
git -C {ROOT} diff --check
git -C {ROOT} status --short
''',PFX+'IMPORT_20260905.txt')
   manifest={}
   with client.open_sftp() as sftp:
    for name in FILES:
     source=sftp.open(ROOT+'/'+name).read();target=LOCAL/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source)
     manifest[name]=hashlib.sha256(source).hexdigest()
    sftp.get(PACKET+'/original-dvac.patch',str(DOC/(PFX+'IMPORTED_DELTA_20260905.patch')))
   (DOC/(PFX+'IMPORT_HASHES_20260905.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
  elif a.mode=='deploy':
   run(f'test "$(git -C {ROOT} rev-parse HEAD)" = {HEAD}')
   manifest=json.loads((DOC/(PFX+'IMPORT_HASHES_20260905.json')).read_text(encoding='utf-8'))
   with client.open_sftp() as sftp:
    for name,digest in manifest.items():assert hashlib.sha256(sftp.open(ROOT+'/'+name).read()).hexdigest()==digest,name
    for name in ['tests/unit_tests/test_online_bc_dvac.py','docs/online_bc_dvac.md',*EXTRA]:
     with sftp.open(ROOT+'/'+name,'wb') as f:f.write((LOCAL/name).read_bytes())
    sftp.put('local_scripts/pi05_bc_dvac_validate_20260905.py',PACKET+'/validate.py')
   run(f'git -C {ROOT} diff --check')
  elif a.mode=='test':
   run(Path('local_scripts/remote_commands/sz_pi05_bc_dvac_validate_20260905.sh').read_text(encoding='utf-8'),PFX+'TEST_CONFIG_20260905.txt')
  elif a.mode=='fetch':
   with client.open_sftp() as sftp:
    for name in [*FILES,*EXTRA]:sftp.get(ROOT+'/'+name,str(LOCAL/name))
    for name,local in [('resolved.yaml',PFX+'FORMAL_RESOLVED_20260905.yaml'),('validation.json',PFX+'VALIDATION_20260905.json')]:sftp.get(PACKET+'/'+name,str(DOC/local))
  elif a.mode=='publish':
   proof=json.loads((DOC/(PFX+'VALIDATION_20260905.json')).read_text(encoding='utf-8'));assert proof['passed']
   assert 'passed' in (DOC/(PFX+'TEST_CONFIG_20260905.txt')).read_text(encoding='utf-8')
   target=ROOT+'/docs/evidence/pi05-online-bc-dvac-20260905'
   names=[PFX+'FORMAL_RESOLVED_20260905.yaml',PFX+'VALIDATION_20260905.json',PFX+'TEST_CONFIG_20260905.txt','GPU7_PI05_BC_DVAC_FORMAL_CONTRACT_20260905.md']
   with client.open_sftp() as sftp:
    sftp.mkdir(target)
    for name in names:
     raw='\n'.join(x.rstrip() for x in (DOC/name).read_text(encoding='utf-8').splitlines())+'\n'
     with sftp.open(target+'/'+name,'wb') as f:f.write(raw.encode())
   paths=[*FILES,*EXTRA,*['docs/evidence/pi05-online-bc-dvac-20260905/'+n for n in names]]
   raw=run(f'''set -eu
cd {ROOT}
test "$(git rev-parse HEAD)" = {HEAD}
git diff --check
git add -f -- {' '.join(paths)}
git diff --cached --check
git diff --cached --stat
git commit -m 'Port existing action DVAC to pi05 BC with mean-one half-range weights'
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 60s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/{BRANCH}
git rev-parse HEAD
git status --porcelain
''',PFX+'PUSH_20260905.txt')
   commit=[x for x in raw.splitlines() if len(x)==40 and all(c in '0123456789abcdef' for c in x)][-1]
   (DOC/(PFX+'SOURCE_HEAD_20260905.txt')).write_text(commit,encoding='utf-8')
  elif a.mode=='launch':
   commit=(DOC/(PFX+'SOURCE_HEAD_20260905.txt')).read_text().strip()
   assert json.loads((DOC/(PFX+'VALIDATION_20260905.json')).read_text())['passed']
   run(f'''set -eu
test "$(git -C {ROOT} rev-parse HEAD)" = {commit}
test -z "$(git -C {ROOT} status --porcelain)"
test ! -e {RUN}
test -z "$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader,nounits)"
df -B1 /data
''')
   with client.open_sftp() as sftp:
    for name in [*FILES,*EXTRA]:assert sftp.open(ROOT+'/'+name).read()==(LOCAL/name).read_bytes(),name
    sftp.mkdir(RUN);sftp.mkdir(RUN+'/runtime')
    for local,name in [(Path('local_scripts/pi05_bc_dvac_formal_wrapper_20260905.sh'),'wrapper.sh'),(DOC/(PFX+'FORMAL_RESOLVED_20260905.yaml'),'resolved.yaml'),(DOC/'GPU7_PI05_BC_DVAC_FORMAL_CONTRACT_20260905.md','contract.md')]:sftp.put(str(local),RUN+'/runtime/'+name)
    observer=Path('local_scripts/bc_gpu6_resource_observer_20260905.py').read_text(encoding='utf-8').replace('GPU6','GPU7').replace('gpu6','gpu7').replace("'-i','6'","'-i','7'")
    with sftp.open(RUN+'/runtime/resource_observer.py','wb') as f:f.write(observer.encode())
   run(f'''set -eu
bash -n {RUN}/runtime/wrapper.sh
test ! -e {RUN}/wrapper.pid
nohup setsid bash {RUN}/runtime/wrapper.sh > {RUN}/wrapper.log 2>&1 < /dev/null & p=$!
printf '%s\n' "$p" > {RUN}/wrapper.pid
nohup /usr/bin/python3 {RUN}/runtime/resource_observer.py "$p" {RUN} > {RUN}/observer.log 2>&1 < /dev/null & o=$!
printf '%s\n' "$o" > {RUN}/observer.pid
printf 'WRAPPER_PID=%s OBSERVER_PID=%s\n' "$p" "$o"
''',PFX+'LAUNCH_20260905.txt')
  elif a.mode=='status':
   source=Path('local_scripts/remote_commands/sz_pi05_bc_status_20260905.sh').read_text(encoding='utf-8').replace('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1',RUN).replace("root='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc'",f"root='{ROOT}'").replace('gpu6','gpu7')
   run(source,PFX+'STARTUP_20260905.json')
  elif a.mode in ('publish-startup','finish-startup'):
   target=ROOT+'/docs/evidence/pi05-online-bc-dvac-20260905'
   uploads={n:DOC/n for n in [PFX+'STARTUP_20260905.json',PFX+'BINDING_20260905.json',PFX+'LAUNCH_20260905.txt',PFX+'IMPLEMENTATION_LEDGER_20260905.md']}
   uploads['pi05_bc_dvac_formal_wrapper_20260905.sh']=Path('local_scripts/pi05_bc_dvac_formal_wrapper_20260905.sh')
   if a.mode=='publish-startup':
    run(f'test -z "$(git -C {ROOT} status --porcelain)"')
   else:
    run(f'test "$(git -C {ROOT} rev-parse HEAD)" = '+(DOC/(PFX+'SOURCE_HEAD_20260905.txt')).read_text().strip())
    state=run(f'git -C {ROOT} status --porcelain --untracked-files=all')
    allowed={'docs/evidence/pi05-online-bc-dvac-20260905/'+n for n in uploads}
    assert all(s[3:] in allowed for s in state.splitlines()),state
   with client.open_sftp() as sftp:
    for name,path in uploads.items():
     raw='\n'.join(x.rstrip() for x in path.read_text(encoding='utf-8').splitlines()).rstrip()+'\n'
     with sftp.open(target+'/'+name,'wb') as f:f.write(raw.encode())
   run(f'''set -eu
cd {ROOT}
git add -f -- {' '.join('docs/evidence/pi05-online-bc-dvac-20260905/'+n for n in uploads)}
git diff --cached --check
git commit -m 'Record direct GPU7 pi05 BC DVAC formal startup without smoke'
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 60s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/{BRANCH}
git rev-parse HEAD
git status --porcelain
''',PFX+'STARTUP_PUSH_20260905.txt')
 finally:
  client.close();os.environ.pop('SEETA_SSH_PASSWORD',None)

if __name__=='__main__':main()
