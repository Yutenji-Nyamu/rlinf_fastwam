"""Capture a reviewed read-only SZ command directly to a local artifact."""
import argparse
import getpass
import os
from pathlib import Path
import remote_exec_autodl

parser=argparse.ArgumentParser()
parser.add_argument('--command-file',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
password=[REDACTED]'SSH password: ')
os.environ['SEETA_SSH_PASSWORD']=password
client=None
try:
    conn=argparse.Namespace(host='120.241.223.9',port=22,user='chenyiteng',host_key_sha256='qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY',timeout=20.0)
    client=remote_exec_autodl.connect(conn)
    stdin,stdout,stderr=client.exec_command(Path(args.command_file).read_text(encoding='utf-8'),timeout=60)
    stdin.channel.shutdown_write()
    data=stdout.read()
    error=stderr.read()
    rc=stdout.channel.recv_exit_status()
    Path(args.output).write_bytes(data)
    Path(args.output+'.err').write_bytes(error)
    print(f'rc={rc} stdout_bytes={len(data)} stderr_bytes={len(error)} output={args.output}')
    raise SystemExit(rc)
finally:
    password=''
    os.environ.pop('SEETA_SSH_PASSWORD',None)
    if client is not None:client.close()
