"""Scoped deployment for the authorized isolated BC-DVAC worktree; no job replay."""
import argparse
import getpass
import os
from pathlib import Path

import remote_exec_autodl as ssh

BASE = '/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc'
ROOT = BASE + '-dvac'
HEAD = '385d4e75bf71cdca5d1ae3a8fc56445ab283185e'
BRANCH = 'codex/sz-pi0-online-bc-dvac'
PACKET = '/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-dvac-20260905'
LOCAL = Path('worktrees/pi0-online-bc-dvac')
FILES = [
    'rlinf/algorithms/online_bc_dvac.py',
    'rlinf/data/online_bc.py',
    'rlinf/models/embodiment/openpi/openpi_action_model.py',
    'rlinf/workers/actor/fsdp_online_bc_policy_worker.py',
    'rlinf/workers/env/env_worker.py',
    'rlinf/workers/rollout/hf/huggingface_worker.py',
    'examples/embodiment/config/bc_dvac/default.yaml',
    'tests/unit_tests/test_online_bc_dvac.py',
    'docs/online_bc_dvac.md',
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','deploy','command','fetch','config-fix','fetch-source','fetch-contract','publish','launch','publish-evidence'])
    p.add_argument('--file')
    p.add_argument('--output')
    p.add_argument('--remote')
    args=p.parse_args()
    os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
    client=ssh.connect(argparse.Namespace(host='120.241.223.9',port=22,user='chenyiteng',timeout=20,
        host_key_sha256='qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'))
    def run(command):
        _,out,err=client.exec_command(command)
        parts=[]
        for line in out:
            parts.append(line)
            print(line.rstrip() if len(line)<12000 else f'[Captured long evidence line: {len(line)} characters; see output file]',flush=True)
        error=err.read().decode(errors='replace')
        if error: print(error,flush=True)
        data=''.join(parts)
        if args.output:
            output=Path(args.output)
            output.write_text(data if output.suffix=='.json' else data+'\n'+error,encoding='utf-8')
        rc=out.channel.recv_exit_status()
        if rc: raise RuntimeError(f'Command exited {rc}; not replayed')
        return data
    try:
        run('id; date -Is')
        if args.mode=='prepare':
            run(f'''set -eu
test "$(git -C {BASE} rev-parse HEAD)" = {HEAD}
test -z "$(git -C {BASE} status --porcelain)"
test ! -e {ROOT}
git -C {BASE} worktree add -b {BRANCH} {ROOT} {HEAD}
mkdir {PACKET}
git -C {ROOT} rev-parse HEAD
git -C {ROOT} status --porcelain
''')
        elif args.mode=='deploy':
            run(f'test "$(git -C {ROOT} rev-parse HEAD)" = {HEAD} && test -z "$(git -C {ROOT} status --porcelain)"')
            with client.open_sftp() as sftp:
                for rel in FILES:
                    sftp.put(str(LOCAL/rel),ROOT+'/'+rel)
                    print('UPLOADED',rel,flush=True)
        elif args.mode=='command':
            run(Path(args.file).read_text(encoding='utf-8'))
        elif args.mode=='config-fix':
            run(f'''set -eu
test "$(git -C {ROOT} branch --show-current)" = {BRANCH}
test "$(git -C {ROOT} rev-parse HEAD)" = {HEAD}
test -f {ROOT}/examples/embodiment/config/robotwin_adjust_bottle_online_bc_dvac_openpi.yaml
mkdir {ROOT}/examples/embodiment/config/bc_dvac
rm -- {ROOT}/examples/embodiment/config/robotwin_adjust_bottle_online_bc_dvac_openpi.yaml
''')
            with client.open_sftp() as sftp:
                for rel in ('examples/embodiment/config/bc_dvac/default.yaml','tests/unit_tests/test_online_bc_dvac.py'):
                    sftp.put(str(LOCAL/rel),ROOT+'/'+rel)
                    print('UPLOADED',rel,flush=True)
        elif args.mode=='fetch-source':
            with client.open_sftp() as sftp:
                for rel in FILES:
                    sftp.get(ROOT+'/'+rel,str(LOCAL/rel))
                for rel in ('method-config-diff.json','inherited-resolved.yaml'):
                    sftp.get(PACKET+'/'+rel,str(Path('docs/rlinf-robotwin-pi0-online-bc/evidence')/('DVAC_'+rel)))
        elif args.mode=='fetch-contract':
            with client.open_sftp() as sftp:
                for remote,local in (
                    ('smoke-resolved.yaml','DVAC_SMOKE_RESOLVED_20260905.yaml'),
                    ('formal-base-to-smoke-diff.json','DVAC_FORMAL_BASE_TO_SMOKE_DIFF_20260905.json'),
                ):
                    sftp.get(PACKET+'/'+remote,'docs/rlinf-robotwin-pi0-online-bc/evidence/'+local)
        elif args.mode=='publish':
            proof=Path('docs/rlinf-robotwin-pi0-online-bc/evidence/DVAC_TEST_CONFIG_RETEST_20260905.txt').read_text(encoding='utf-8')
            assert '22 passed' in proof and 'TEST_AND_CONFIG_VALIDATION_PASSED' in proof
            run(f'test "$(git -C {ROOT} rev-parse HEAD)" = {HEAD}; git -C {ROOT} diff --check')
            evidence=ROOT+'/docs/evidence/online-bc-dvac-20260905'
            run(f'mkdir {evidence}')
            with client.open_sftp() as sftp:
                for name in ('DVAC_TEST_CONFIG_RETEST_20260905.txt','DVAC_SMOKE_RESOLVED_20260905.yaml','DVAC_FORMAL_BASE_TO_SMOKE_DIFF_20260905.json','GPU7_DVAC_SMOKE_CONTRACT_20260905.md'):
                    sftp.put('docs/rlinf-robotwin-pi0-online-bc/evidence/'+name,evidence+'/'+name)
            # Explicit staged allowlist; no unrelated dirty or runtime files.
            names=' '.join(FILES)+ ' docs/evidence/online-bc-dvac-20260905'
            run(f'''set -eu
cd {ROOT}
git diff --check
git add -- {names}
git diff --cached --stat
git commit -m 'Add opt-in action-level DVAC to online success flow BC'
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 90s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/{BRANCH}
git rev-parse HEAD
git status --porcelain
''')
        elif args.mode=='publish-evidence':
            import json
            evidence=ROOT+'/docs/evidence/online-bc-dvac-20260905'
            local_evidence=Path('docs/rlinf-robotwin-pi0-online-bc/evidence')
            proof=json.loads((local_evidence/'DVAC_SMOKE_VERIFICATION_20260905.json').read_text(encoding='utf-8'))
            assert proof['passed'] and proof['optimizer_updates']==20
            names=('DVAC_TEST_CONFIG_RETEST_20260905.txt','DVAC_SMOKE_VERIFICATION_20260905.json','DVAC_SMOKE_STATUS_20260905.json','DVAC_SMOKE_RESULT_FOR_GIT_20260905.md')
            run(f'test "$(git -C {ROOT} rev-parse HEAD)" = 736b1416f37f32034efd3924a0fc5f5fca611012')
            state=run(f'git -C {ROOT} status --porcelain --untracked-files=all')
            allowed={'docs/evidence/online-bc-dvac-20260905/'+n for n in (*names,'success.png','resources.png')}
            assert all(line[3:] in allowed for line in state.splitlines()), 'Unrelated dirty path; not publishing.'
            with client.open_sftp() as sftp:
                for name in names:
                    assert (local_evidence/name).stat().st_size<2_000_000
                    # Normalize the publication copy only; retain raw Windows logs locally.
                    value=(local_evidence/name).read_text(encoding='utf-8')
                    value='\n'.join(line.rstrip() for line in value.splitlines())+'\n'
                    with sftp.open(evidence+'/'+name,'wb') as handle:handle.write(value.encode('utf-8'))
                for local,rel in ((local_evidence/'bc-dvac-status-20260905/success.png','success.png'),(local_evidence/'bc-dvac-status-20260905/resources.png','resources.png')):
                    sftp.put(str(local),evidence+'/'+rel)
            allowlist=' '.join('docs/evidence/online-bc-dvac-20260905/'+n for n in (*names,'success.png','resources.png'))
            run(f'''set -eu
cd {ROOT}
git add -f -- {allowlist}
git diff --cached --check
git diff --cached --stat
git commit -m 'Record two-round online BC DVAC smoke verification and resource evidence'
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 90s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/{BRANCH}
git rev-parse HEAD
git status --porcelain
''')
        elif args.mode=='launch':
            import hashlib
            run_dir='/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1'
            run(f'''set -eu
test "$(git -C {ROOT} branch --show-current)" = {BRANCH}
test -z "$(git -C {ROOT} status --porcelain)"
test ! -e {run_dir}
test -z "$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader,nounits)"
df -B1 /data
git -C {BASE} rev-parse HEAD
git -C {BASE} status --porcelain
''')
            with client.open_sftp() as sftp:
                for rel in FILES:
                    assert sftp.open(ROOT+'/'+rel).read()==(LOCAL/rel).read_bytes(),rel
                assert sftp.open(PACKET+'/smoke-resolved.yaml').read()==Path('docs/rlinf-robotwin-pi0-online-bc/evidence/DVAC_SMOKE_RESOLVED_20260905.yaml').read_bytes()
                sftp.mkdir(run_dir);sftp.mkdir(run_dir+'/runtime')
                for local,remote in (
                    ('local_scripts/bc_dvac_smoke_wrapper_20260905.sh','wrapper.sh'),
                    ('docs/rlinf-robotwin-pi0-online-bc/evidence/DVAC_SMOKE_RESOLVED_20260905.yaml','resolved.yaml'),
                    ('docs/rlinf-robotwin-pi0-online-bc/evidence/GPU7_DVAC_SMOKE_CONTRACT_20260905.md','contract.md'),
                ):
                    sftp.put(local,run_dir+'/runtime/'+remote)
                # Mechanically reuse the already-tested observer on GPU7 only.
                observer=Path('local_scripts/bc_gpu6_resource_observer_20260905.py').read_text(encoding='utf-8').replace('GPU6','GPU7').replace('gpu6','gpu7').replace("'-i','6'","'-i','7'")
                with sftp.open(run_dir+'/runtime/resource_observer.py','wb') as f:f.write(observer.encode())
            run(f'nohup setsid bash {run_dir}/runtime/wrapper.sh > {run_dir}/wrapper.log 2>&1 < /dev/null & p=$!; printf "%s\\n" "$p" > {run_dir}/wrapper.pid; nohup /usr/bin/python3 {run_dir}/runtime/resource_observer.py "$p" {run_dir} > {run_dir}/observer.log 2>&1 < /dev/null & printf "WRAPPER_PID=%s OBSERVER_PID=%s\\n" "$p" "$!"')
        else:
            with client.open_sftp() as sftp:
                sftp.get(args.remote,args.file)
    finally:
        client.close()
        os.environ.pop('SEETA_SSH_PASSWORD',None)


if __name__=='__main__': main()
