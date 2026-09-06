"""Exact two-file deployment and bounded GPU6 smoke; no formal launch mode."""
import argparse
import getpass
import os
from pathlib import Path
import remote_exec_autodl as ssh

ROOT='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc'
HEAD='385d4e75bf71cdca5d1ae3a8fc56445ab283185e'
PACKET='/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-eval8-20260905'
RUN='/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval8x4-gpu6-20260905-v8'
FILES=('examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml','tests/unit_tests/test_online_bc.py')
LOCAL=Path('docs/rlinf-robotwin-pi0-online-bc/evidence')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('deploy','fetch','launch','publish-result'))
    args=p.parse_args()
    os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
    client=ssh.connect(argparse.Namespace(host='120.241.223.9',port=22,user='chenyiteng',timeout=20,host_key_sha256='qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'))
    def command(text):
        _,out,err=client.exec_command(text)
        data=out.read().decode(errors='replace');error=err.read().decode(errors='replace')
        print(data,flush=True)
        if error:print(error,flush=True)
        assert out.channel.recv_exit_status()==0,'Scoped command failed; not replayed.'
        return data
    try:
        command('id; date -Is')
        if args.mode=='deploy':
            command(f'test "$(git -C {ROOT} rev-parse HEAD)" = {HEAD} && test -z "$(git -C {ROOT} status --porcelain)" && test ! -e {PACKET}')
            with client.open_sftp() as sftp:
                sftp.mkdir(PACKET)
                for rel in FILES:sftp.put(str(Path('worktrees/pi0-online-bc')/rel),ROOT+'/'+rel)
        elif args.mode=='fetch':
            with client.open_sftp() as sftp:
                for rel in FILES:sftp.get(ROOT+'/'+rel,str(Path('worktrees/pi0-online-bc')/rel))
                for src,dest in (('smoke-resolved.yaml','BC_EVAL8_SMOKE_RESOLVED_20260905.yaml'),('diff.json','BC_EVAL8_DIFF_20260905.json'),('fixed-seeds.json','BC_EVAL8_FIXED_SEEDS_20260905.json')):
                    sftp.get(PACKET+'/'+src,str(LOCAL/dest))
        elif args.mode=='launch':
            command(f'test -z "$(git -C {ROOT} status --porcelain)" && test ! -e {RUN} && test -z "$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader,nounits)"')
            with client.open_sftp() as sftp:
                for rel in FILES:assert sftp.open(ROOT+'/'+rel).read()==(Path('worktrees/pi0-online-bc')/rel).read_bytes()
                assert sftp.open(PACKET+'/smoke-resolved.yaml').read()==(LOCAL/'BC_EVAL8_SMOKE_RESOLVED_20260905.yaml').read_bytes()
                sftp.mkdir(RUN);sftp.mkdir(RUN+'/runtime')
                for src,dest in (('local_scripts/bc_eval8_wrapper_20260905.sh','wrapper.sh'),('local_scripts/bc_gpu6_resource_observer_20260905.py','resource_observer.py'),(str(LOCAL/'BC_EVAL8_SMOKE_RESOLVED_20260905.yaml'),'resolved.yaml'),(str(LOCAL/'GPU6_EVAL8_SMOKE_CONTRACT_20260905.md'),'contract.md')):
                    sftp.put(src,RUN+'/runtime/'+dest)
            command(f'nohup setsid bash {RUN}/runtime/wrapper.sh > {RUN}/wrapper.log 2>&1 < /dev/null & p=$!; printf "%s\\n" "$p" > {RUN}/wrapper.pid; nohup /usr/bin/python3 {RUN}/runtime/resource_observer.py "$p" {RUN} > {RUN}/observer.log 2>&1 < /dev/null & printf "WRAPPER_PID=%s OBSERVER_PID=%s\\n" "$p" "$!"')
        elif args.mode=='publish-result':
            import json
            proof=json.loads((LOCAL/'BC_EVAL8_SMOKE_VERIFICATION_20260905.json').read_text(encoding='utf-8'))
            assert proof['passed'] and proof['optimizer_updates']==20
            target=ROOT+'/docs/evidence/online-bc-eval8-20260905'
            command(f'test "$(git -C {ROOT} rev-parse HEAD)" = a8764944763f12b8a2bfa1ba58c828192326ac30 && test -z "$(git -C {ROOT} status --porcelain)"')
            names=('BC_EVAL8_TEST_CONFIG_20260905.txt','BC_EVAL8_SMOKE_VERIFICATION_20260905.json','BC_EVAL8_SMOKE_RESOLVED_20260905.yaml','BC_EVAL8_DIFF_20260905.json','BC_EVAL8_FIXED_SEEDS_20260905.json','GPU6_EVAL8_SMOKE_CONTRACT_20260905.md')
            with client.open_sftp() as sftp:
                sftp.mkdir(target)
                for n in names:
                    value='\n'.join(line.rstrip() for line in (LOCAL/n).read_text(encoding='utf-8').splitlines())+'\n'
                    with sftp.open(target+'/'+n,'wb') as handle:handle.write(value.encode('utf-8'))
            command(f'''set -eu
cd {ROOT}
git add -f -- {' '.join('docs/evidence/online-bc-eval8-20260905/'+n for n in names)}
git diff --cached --check
git commit -m 'Record eval8x4 GPU6 BC smoke verification'
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 90s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/codex/sz-pi0-online-bc
git rev-parse HEAD
git status --porcelain
''')
    finally:
        client.close();os.environ.pop('SEETA_SSH_PASSWORD',None)

if __name__=='__main__':main()
