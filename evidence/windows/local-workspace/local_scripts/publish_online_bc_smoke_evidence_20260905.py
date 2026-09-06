"""Publish ended owned smoke evidence and the specific fd reproduction only."""
import getpass
import os
from pathlib import Path
import remote_exec_autodl as ssh

args=ssh.build_parser().parse_args(['--host','120.241.223.9','--port','22','--user','chenyiteng','--host-key-sha256','qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY','run','--stdin-file','local_scripts/backfill_bc_smoke_evidence_20260905.py','/usr/bin/python3 -'])
os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
client=ssh.connect(args)
try:
    root='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc'
    _,out,err=client.exec_command(f'id; git -C {root} rev-parse HEAD; git -C {root} status --porcelain')
    lines=out.read().decode().splitlines()
    assert len(lines)==2 and lines[1]=='72a926041867cbdbf2565ab66a14d742c59a0dad',lines
    print('\n'.join(lines),flush=True)
    stage='/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-20260905/fd-evidence-inputs'
    with client.open_sftp() as sftp:
        sftp.mkdir(stage)
        for path in ('local_scripts/bc_env_fd_probe_20260905.py','docs/rlinf-robotwin-pi0-online-bc/evidence/BC_FD_EXHAUSTION_PROBE_20260905.md'):
            sftp.put(path,stage+'/'+Path(path).name)
    raise SystemExit(ssh.run_command(client,args))
finally:
    client.close()
    os.environ.pop('SEETA_SSH_PASSWORD',None)
