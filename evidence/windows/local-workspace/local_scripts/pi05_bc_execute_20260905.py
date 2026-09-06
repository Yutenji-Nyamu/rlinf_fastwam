"""Scoped pi05 BC worktree/setup/transfer; never launches formal training."""
import argparse
import getpass
import json
import os
from pathlib import Path

import remote_exec_autodl as ssh

BASE = '/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc'
ROOT = '/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc'
SIDNEY = '/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf'
HEAD = '2467d997831166b70444b0c99d5198a2d3dfc8f6'
BRANCH = 'codex/sz-pi05-online-bc'
MODEL = '/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab'
PACKET = '/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-pi05-20260905'
RUN = '/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1'
LOCAL = Path('worktrees/pi05-online-bc')
DOC = Path('docs/rlinf-robotwin-pi0-online-bc/evidence')
FILES = ('rlinf/models/embodiment/openpi/dataconfig/__init__.py',
         'rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py',
         'examples/embodiment/config/online_bc_model/pi05_sidney.yaml',
         'tests/unit_tests/test_pi05_online_bc.py')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare','deploy','config-fix','update','data-check','command','fetch','launch','watch','publish-evidence'))
    p.add_argument('--file'); p.add_argument('--output')
    args=p.parse_args()
    os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
    client=ssh.connect(argparse.Namespace(host='120.241.223.9',port=22,user='chenyiteng',timeout=20,
        host_key_sha256='qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'))
    def run(command, output=None):
        _,out,err=client.exec_command(command)
        data=out.read().decode(errors='replace'); error=err.read().decode(errors='replace')
        rc=out.channel.recv_exit_status()
        if output:
            Path(output).write_text(data if str(output).endswith('.json') else data+'\n'+error,encoding='utf-8')
            if error and str(output).endswith('.json'): Path(output).with_suffix('.stderr.txt').write_text(error,encoding='utf-8')
        print(data[-10000:],flush=True)
        if error: print(error[-4000:],flush=True)
        assert rc==0, 'Scoped command failed; never automatically replay a side effect.'
        return data
    try:
        run('id; date -Is')
        if args.mode=='prepare':
            assert not LOCAL.exists(), 'Local staging exists; inspect before reuse.'
            run(f'''set -eu
test "$(git -C {BASE} rev-parse HEAD)" = {HEAD}
test -z "$(git -C {BASE} status --porcelain)"
test -z "$(git -C {SIDNEY} status --porcelain)"
test ! -e {ROOT}
test ! -e {PACKET}
test ! -e {RUN}
test -f {MODEL}/model.safetensors
test -z "$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader,nounits)"
if git -C {BASE} show-ref --verify --quiet refs/heads/{BRANCH}; then exit 91; fi
git -C {SIDNEY} rev-parse HEAD
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
df -B1 /data /home
git -C {BASE} worktree add -b {BRANCH} {ROOT} {HEAD}
mkdir {PACKET}
''',DOC/'PI05_BC_PREPARE_20260905.txt')
            with client.open_sftp() as sftp:
                for name in FILES[:2]:
                    target=LOCAL/name; target.parent.mkdir(parents=True,exist_ok=True)
                    sftp.get(ROOT+'/'+name,str(target))
                for name in ('examples/embodiment/config/model/pi0_5.yaml',
                             'examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml',
                             'rlinf/envs/robotwin/robotwin_env.py'):
                    target=LOCAL/name; target.parent.mkdir(parents=True,exist_ok=True)
                    sftp.get(ROOT+'/'+name,str(target))
        elif args.mode in ('deploy','config-fix','update'):
            run(f'test "$(git -C {ROOT} rev-parse HEAD)" = {HEAD}')
            if args.mode=='deploy': run(f'test -z "$(git -C {ROOT} status --porcelain)"')
            with client.open_sftp() as sftp:
                if args.mode=='config-fix':
                    obsolete=ROOT+'/examples/embodiment/config/robotwin_move_pillbottle_online_bc_openpi_pi05.yaml'
                    assert sftp.open(obsolete).read().startswith(b'# Sidney model/task adapter;')
                    sftp.remove(obsolete)  # Only our uncommitted failed composition, not user data.
                    sftp.mkdir(ROOT+'/examples/embodiment/config/online_bc_model')
                for name in FILES:
                    raw=(LOCAL/name).read_bytes()
                    with sftp.open(ROOT+'/'+name,'wb') as f:f.write(raw)
                for name in ('pi05_bc_validate_20260905.py',):
                    sftp.put('local_scripts/'+name,PACKET+'/'+name)
            run(f'git -C {ROOT} diff --check; git -C {ROOT} status --short; git -C {ROOT} diff --stat')
        elif args.mode=='data-check':
            with client.open_sftp() as sftp:
                sftp.put('local_scripts/pi05_bc_validate_20260905.py',PACKET+'/pi05_bc_validate_20260905.py')
            setup=Path('local_scripts/remote_commands/sz_pi05_bc_validate_20260905.sh').read_text(encoding='utf-8').split('cd "$root"',1)[0]
            run(setup+'cd "$root"\n"$venv/bin/python" -u '+PACKET+'/pi05_bc_validate_20260905.py\n',DOC/'PI05_BC_DATA_VALIDATION_RETEST_20260905.txt')
        elif args.mode=='publish-evidence':
            proof=json.loads((DOC/'PI05_BC_SMOKE_VERIFICATION_20260905.json').read_text(encoding='utf-8'))
            assert proof['passed'] and proof['optimizer_updates']==20
            run(f'test "$(git -C {ROOT} rev-parse HEAD)" = 653fe0fbc05188eb0ec19077de5c78a00b8230ad && test -z "$(git -C {ROOT} status --porcelain)"')
            names=('PI05_BC_SMOKE_VERIFICATION_20260905.json','PI05_BC_SMOKE_RESOLVED_20260905.yaml',
                   'PI05_BC_RESOLVED_DELTA_20260905.json','PI05_BC_VALIDATION_20260905.json',
                   'PI05_BC_TEST_CONFIG_FINAL_20260905.txt','PI05_BC_DATA_VALIDATION_RETEST_20260905.txt',
                   'GPU6_PI05_BC_SMOKE_CONTRACT_20260905.md','PI05_BC_EVIDENCE_README_20260905.md')
            scripts=('pi05_bc_wrapper_20260905.sh','bc_gpu6_resource_observer_20260905.py',
                     'pi05_bc_validate_20260905.py','remote_commands/sz_pi05_bc_verify_20260905.sh')
            uploads={name:DOC/name for name in names}
            uploads.update({Path(name).name:Path('local_scripts')/name for name in scripts})
            target=ROOT+'/docs/evidence/pi05-online-bc-smoke-20260905'
            with client.open_sftp() as sftp:
                sftp.mkdir(target)
                for name,source in uploads.items():
                    raw=source.read_text(encoding='utf-8')
                    raw='\n'.join(line.rstrip() for line in raw.splitlines())+'\n'
                    assert len(raw.encode())<1_000_000
                    with sftp.open(target+'/'+name,'wb') as handle:handle.write(raw.encode())
            run(f'''set -eu
cd {ROOT}
git add -f -- {' '.join('docs/evidence/pi05-online-bc-smoke-20260905/'+n for n in uploads)}
git diff --cached --check
git commit -m 'Record pi05 online BC two-round GPU6 smoke validation'
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 90s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/{BRANCH}
git rev-parse HEAD
git status --porcelain
''',DOC/'PI05_BC_EVIDENCE_PUSH_20260905.txt')
        elif args.mode=='watch':
            _,out,err=client.exec_command(Path('local_scripts/remote_commands/sz_pi05_bc_watch_20260905.sh').read_text(encoding='utf-8'))
            with (DOC/'PI05_BC_WATCH_20260905.jsonl').open('a',encoding='utf-8') as log:
                for line in out:
                    print(line,end='',flush=True);log.write(line);log.flush()
            error=err.read().decode(errors='replace')
            if error:print(error,flush=True)
            assert out.channel.recv_exit_status()==0
        elif args.mode=='command':
            run(Path(args.file).read_text(encoding='utf-8'),args.output)
        elif args.mode=='fetch':
            with client.open_sftp() as sftp:
                for name in FILES: sftp.get(ROOT+'/'+name,str(LOCAL/name))
                seed='rlinf/envs/robotwin/seeds/eval_sidney_fixed32.json'
                (LOCAL/seed).parent.mkdir(parents=True,exist_ok=True)
                sftp.get(ROOT+'/'+seed,str(LOCAL/seed))
                for name,dest in (('smoke-resolved.yaml','PI05_BC_SMOKE_RESOLVED_20260905.yaml'),
                                  ('delta.json','PI05_BC_RESOLVED_DELTA_20260905.json'),
                                  ('validation.json','PI05_BC_VALIDATION_20260905.json')):
                    sftp.get(PACKET+'/'+name,str(DOC/dest))
        elif args.mode=='launch':
            run(f'''set -eu
test "$(git -C {ROOT} branch --show-current)" = {BRANCH}
test -z "$(git -C {ROOT} status --porcelain)"
test ! -e {RUN}
test -z "$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader,nounits)"
''')
            with client.open_sftp() as sftp:
                for name in (*FILES,'rlinf/envs/robotwin/seeds/eval_sidney_fixed32.json'): assert sftp.open(ROOT+'/'+name).read()==(LOCAL/name).read_bytes(),name
                assert sftp.open(PACKET+'/smoke-resolved.yaml').read()==(DOC/'PI05_BC_SMOKE_RESOLVED_20260905.yaml').read_bytes()
                sftp.mkdir(RUN); sftp.mkdir(RUN+'/runtime')
                for source,name in (('local_scripts/pi05_bc_wrapper_20260905.sh','wrapper.sh'),
                                    ('local_scripts/bc_gpu6_resource_observer_20260905.py','resource_observer.py'),
                                    (str(DOC/'PI05_BC_SMOKE_RESOLVED_20260905.yaml'),'resolved.yaml'),
                                    (str(DOC/'GPU6_PI05_BC_SMOKE_CONTRACT_20260905.md'),'contract.md')):
                    sftp.put(source,RUN+'/runtime/'+name)
            run(f'''nohup setsid bash {RUN}/runtime/wrapper.sh > {RUN}/wrapper.log 2>&1 < /dev/null & p=$!
printf '%s\n' "$p" > {RUN}/wrapper.pid
nohup /usr/bin/python3 {RUN}/runtime/resource_observer.py "$p" {RUN} > {RUN}/observer.log 2>&1 < /dev/null &
printf 'WRAPPER_PID=%s OBSERVER_PID=%s\n' "$p" "$!"
''',DOC/'PI05_BC_LAUNCH_20260905.txt')
    finally:
        client.close(); os.environ.pop('SEETA_SSH_PASSWORD',None)

if __name__=='__main__': main()
