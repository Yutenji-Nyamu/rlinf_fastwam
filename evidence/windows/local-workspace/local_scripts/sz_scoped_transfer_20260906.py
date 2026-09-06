"""Explicit file transfer and a single command; fixed host, no command retries."""
import argparse,getpass,json,os,posixpath
from pathlib import Path
import remote_exec_autodl as ssh
p=argparse.ArgumentParser();p.add_argument('--uploads');p.add_argument('--command',required=True);p.add_argument('--output',required=True);p.add_argument('--secret-stdin',action='store_true')
a=p.parse_args();assert not Path(a.output).exists()
os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
c=ssh.connect(argparse.Namespace(host='120.241.223.9',port=22,user='chenyiteng',timeout=20,host_key_sha256='qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'))
try:
 _,out,err=c.exec_command('id; date -Is');print(out.read().decode(),flush=True);assert out.channel.recv_exit_status()==0
 if a.uploads:
  with c.open_sftp() as s:
   for item in json.loads(Path(a.uploads).read_text(encoding='utf-8')):
    dest=item['remote'];assert dest.startswith('/data/chenyiteng/results/server-maintenance-20260906/') and '..' not in Path(dest).parts
    parent=posixpath.dirname(dest);parts=parent.split('/');current=''
    for part in parts:
     if not part:continue
     current+='/'+part
     try:s.stat(current)
     except FileNotFoundError:s.mkdir(current)
    try:s.stat(dest)
    except FileNotFoundError:pass
    else:raise RuntimeError('Do not overwrite remote packet '+dest)
    progress=[0]
    def transferred(done,total):
     if done-progress[0]>=50*1024**2:
      progress[0]=done;print('upload progress '+str(done//1024**2)+'/'+str(total//1024**2)+' MiB',flush=True)
    s.put(item['local'],dest,callback=transferred);print('uploaded '+dest,flush=True)
 inp,out,err=c.exec_command(Path(a.command).read_text(encoding='utf-8'))
 if a.secret_stdin:
  inp.write(os.environ['SEETA_SSH_PASSWORD']+'\n');inp.flush();inp.channel.shutdown_write()
 with Path(a.output).open('x',encoding='utf-8') as log:
  for line in out:log.write(line);log.flush();print(line.rstrip()[:1500],flush=True)
 error=err.read().decode(errors='replace')
 if error:Path(a.output).with_suffix('.stderr.txt').write_text(error,encoding='utf-8');print(error[-3000:],flush=True)
 assert out.channel.recv_exit_status()==0,'Command failed, do not automatically replay'
finally:c.close();os.environ.pop('SEETA_SSH_PASSWORD',None)
