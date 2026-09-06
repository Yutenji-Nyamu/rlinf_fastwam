"""One pinned password session; no automatic replay of command side effects."""
import argparse, concurrent.futures, getpass, json, os
from pathlib import Path
import remote_exec_autodl as ssh

p=argparse.ArgumentParser()
p.add_argument('--jobs',required=True,help='JSON pairs of local command file and new output file')
args=p.parse_args()
jobs=json.loads(Path(args.jobs).read_text(encoding='utf-8'))
for command,dest in jobs:
    assert Path(command).is_file()
    assert not Path(dest).exists(), f'Preserve frozen evidence: {dest}'
os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
c=ssh.connect(argparse.Namespace(host='120.241.223.9',port=22,user='chenyiteng',timeout=20,
    host_key_sha256='qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'))
try:
    _,so,se=c.exec_command('id; date -Is')
    print(so.read().decode(),flush=True);assert so.channel.recv_exit_status()==0
    def run(job):
        command,dest=job
        _,so,se=c.exec_command(Path(command).read_text(encoding='utf-8'))
        data=so.read().decode(errors='replace');err=se.read().decode(errors='replace')
        rc=so.channel.recv_exit_status()
        Path(dest).write_text(data,encoding='utf-8')
        if err:Path(dest).with_suffix('.stderr.txt').write_text(err,encoding='utf-8')
        assert rc==0,(command,rc,err[-1500:])
        return {'saved':dest,'bytes':len(data.encode()),'rc':rc}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
finally:
    c.close();os.environ.pop('SEETA_SSH_PASSWORD',None)
