"""Launch exactly the displayed GPU6 BC smoke; preserve all other processes."""
import getpass
import argparse
import hashlib
import json
import os
from pathlib import Path

import remote_exec_autodl as ssh

ROOT='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc'
RUN='/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v6'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--formal',action='store_true')
    cli=parser.parse_args()
    global RUN
    if cli.formal:
        RUN='/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-online-bc32x1-m4-gpu6-formal100-20260905-v1'
    args=ssh.build_parser().parse_args(['--host','120.241.223.9','--port','22','--user','chenyiteng','--host-key-sha256','qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY','run','true'])
    os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
    client=ssh.connect(args)
    try:
        command=f"id; git -C {ROOT} rev-parse HEAD; git -C {ROOT} branch --show-current; nvidia-smi -i 6 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits; test ! -e {RUN}"
        _,out,err=client.exec_command(command)
        raw=out.read().decode(); error=err.read().decode(); rc=out.channel.recv_exit_status()
        print(raw,flush=True)
        if rc or 'codex/sz-pi0-online-bc' not in raw: raise RuntimeError(f'Preflight failed: {rc} {error}')
        with client.open_sftp() as sftp:
            if cli.formal:
                smoke='/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v6'
                assert sftp.open(smoke+'/exit_code.txt').read().strip()==b'0'
                # Full metrics/checkpoint review is also performed before this call.
                sftp.stat(smoke+'/finished_at.txt')
            probe='/data/chenyiteng/results/rlinf-shenzhen/online-bc/sft-leaf-wrap-nativeopt-local-orig-false-20260905/result.json'
            proof=json.loads(sftp.open(probe).read())
            assert proof['passed'] and proof['optimizer_updates']==2 and proof['checkpoint_restore_equal']
            sftp.get(probe,'docs/rlinf-robotwin-pi0-online-bc/evidence/SFT_SYNC_CHECKPOINT_PROBE_20260905.json')
            # No changed source since full config validation.
            rel='examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml'
            actual=sftp.open(f'{ROOT}/{rel}').read()
            assert actual == (Path('worktrees/pi0-online-bc')/rel).read_bytes()
            sftp.mkdir(RUN)
            sftp.mkdir(RUN+'/runtime')
            for local,remote in [
                ('local_scripts/bc_gpu6_formal_wrapper_20260905.sh' if cli.formal else 'local_scripts/bc_gpu6_smoke_wrapper_20260905.sh','runtime/wrapper.sh'),
                ('local_scripts/bc_gpu6_resource_observer_20260905.py','runtime/resource_observer.py'),
                ('docs/rlinf-robotwin-pi0-online-bc/evidence/GPU6_FORMAL_RESOLVED_20260905.yaml' if cli.formal else 'docs/rlinf-robotwin-pi0-online-bc/evidence/GPU6_SMOKE_RESOLVED_20260905.yaml','runtime/resolved.yaml'),
                ('docs/rlinf-robotwin-pi0-online-bc/evidence/GPU6_FORMAL_CONTRACT_20260905.md' if cli.formal else 'docs/rlinf-robotwin-pi0-online-bc/evidence/GPU6_SMOKE_CONTRACT_20260905.md','runtime/contract.md')]:
                sftp.put(local,RUN+'/'+remote)
        command=f"nohup setsid bash {RUN}/runtime/wrapper.sh > {RUN}/wrapper.log 2>&1 < /dev/null & p=$!; printf '%s\\n' \"$p\" > {RUN}/wrapper.pid; nohup /usr/bin/python3 {RUN}/runtime/resource_observer.py \"$p\" {RUN} > {RUN}/observer.log 2>&1 < /dev/null & printf 'WRAPPER_PID=%s OBSERVER_PID=%s\\n' \"$p\" \"$!\""
        _,out,err=client.exec_command(command)
        print(out.read().decode(),flush=True)
        error=err.read().decode(); rc=out.channel.recv_exit_status()
        if rc: raise RuntimeError(f'Launch returned {rc}: {error}; inspect before any retry')
    finally:
        client.close(); os.environ.pop('SEETA_SSH_PASSWORD',None)

if __name__=='__main__': main()
