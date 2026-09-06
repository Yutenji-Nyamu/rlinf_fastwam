"""Read only two relevant server reference trees; no history-wide traversal."""
import getpass
import json
import os
from pathlib import Path
import remote_exec_autodl as ssh

args=ssh.build_parser().parse_args(['--host','120.241.223.9','--port','22','--user','chenyiteng','--host-key-sha256','qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY','run','true'])
os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
client=ssh.connect(args)
base=Path('docs/rlinf-robotwin-pi0-online-bc/evidence/dvac-reference-20260905')
try:
 with client.open_sftp() as sftp:
  for name,root,files in [
   ('rlt','/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421', ['rlinf/algorithms/dvac_weighting.py','rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py']),
   ('bc','/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc', ['rlinf/hybrid_engines/fsdp/strategy/fsdp.py','rlinf/hybrid_engines/fsdp/utils.py']),
  ]:
   _,out,err=client.exec_command(f'git -C {root} rev-parse HEAD; git -C {root} status --short')
   lock=out.read().decode(); assert out.channel.recv_exit_status()==0,err.read().decode()
   target=base/name; target.mkdir(parents=True,exist_ok=True)
   (target/'source-lock.txt').write_text(root+'\n'+lock)
   print(name,lock.strip(),flush=True)
   _,tree,treeerr=client.exec_command(f'git -C {root} ls-tree -r --name-only HEAD rlinf')
   tree_paths=tree.read().decode().splitlines()
   assert tree.channel.recv_exit_status()==0,treeerr.read().decode()
   for requested in files:
    matches=[p for p in tree_paths if Path(p).name==Path(requested).name]
    if len(matches)!=1:
     print('LOCATE',requested,[p for p in tree_paths if 'dvac' in p or 'fsdp_rlt' in p],flush=True)
     continue
    rel=matches[0]
    local=target/rel; local.parent.mkdir(parents=True,exist_ok=True)
    sftp.get(root+'/'+rel,str(local)); print('READ',rel,flush=True)
finally:
 client.close(); os.environ.pop('SEETA_SSH_PASSWORD',None)
