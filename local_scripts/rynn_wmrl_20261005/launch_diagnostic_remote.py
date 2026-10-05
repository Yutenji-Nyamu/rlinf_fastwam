import os,socket,subprocess
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
subprocess.run(['/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python','-u','-B',
    '/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-diagnosis-v1/code/rynn_diagnostic_owner.py','launch'],check=True,timeout=90)
