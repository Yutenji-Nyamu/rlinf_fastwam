import json,os,socket,subprocess,hashlib
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
P=Path('/data/chenyiteng/projects/RynnValue-10e0d333');f='rynn_infer/inference.py'
def git(*args):return subprocess.check_output(['git','-C',str(P),*args],text=True)
head=git('show','HEAD:'+f)
print(json.dumps({'head':git('rev-parse','HEAD').strip(),'status':git('status','--short'),
    'diff':git('diff','HEAD','--',f),'head_source':head,
    'worktree_sha256':hashlib.sha256((P/f).read_bytes()).hexdigest(),
    'head_sha256':hashlib.sha256(head.encode()).hexdigest()}))
