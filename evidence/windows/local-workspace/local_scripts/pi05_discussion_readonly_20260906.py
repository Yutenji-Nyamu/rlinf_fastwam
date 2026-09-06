"""Fixed-host read-only batch, with separate local evidence outputs."""
import argparse, concurrent.futures, getpass, json, os
from pathlib import Path
import remote_exec_autodl as ssh

jobs=[('local_scripts/remote_commands/sz_pi05_three_runs_readonly_20260906.sh','docs/rlinf-robotwin-pi0-online-bc/evidence/PI05_DISCUSSION_LIVE_20260906.json'),
      ('local_scripts/remote_commands/sz_pi05_seeds_readonly_20260906.sh','docs/rlinf-robotwin-pi0-online-bc/evidence/PI05_SEEDS_SOURCE_20260906.json'),
      ('local_scripts/remote_commands/sz_own_storage_git_readonly_20260906.sh','docs/server-admin/SZ_STORAGE_GIT_READONLY_20260906.json')]
os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
c=ssh.connect(argparse.Namespace(host='120.241.223.9',port=22,user='chenyiteng',timeout=20,
    host_key_sha256='qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'))
try:
    _,so,se=c.exec_command('id; date -Is');print(so.read().decode(),flush=True);assert so.channel.recv_exit_status()==0
    def run(job):
        file,dest=job;_,so,se=c.exec_command(Path(file).read_text(encoding='utf-8'))
        data=so.read().decode(errors='replace');err=se.read().decode(errors='replace');rc=so.channel.recv_exit_status()
        Path(dest).write_text(data,encoding='utf-8')
        if err:Path(dest).with_suffix('.stderr.txt').write_text(err,encoding='utf-8')
        assert rc==0,(file,rc,err[-1000:]);obj=json.loads(data)
        return {'saved':dest,'bytes':len(data.encode()),'keys':list(obj)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:
        for f in concurrent.futures.as_completed([p.submit(run,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
finally:c.close();os.environ.pop('SEETA_SSH_PASSWORD',None)
