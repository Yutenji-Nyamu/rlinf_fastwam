"""Read only the exact restored RLT45 runs and protected GPU67 identities."""
import json, os, socket, subprocess
from pathlib import Path
from datetime import datetime
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
F=S/'rlt45-return-20261007-v1'
read=lambda p:json.loads(p.read_text())
def proc(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return {'pid':pid,'alive':False}
    return {'pid':pid,'alive':True,'uid':p.stat().st_uid,'start':int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),'argv':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
out={'time':datetime.now().astimezone().isoformat(),'runs':{},'protected':[proc(x) for x in [3182779,3182804,3651109]],'dispatch':read(F/'dispatched.json')}
for gpu in [4,5]:
    c=F/f'return-g{gpu}';plan=read(c/'plan.json');key=f'gpu{gpu}';row=plan['runs'][key];r=Path(row['new_run'])
    d={'run':str(r),'namespace':row['namespace'],'checkpoint':row['recovery']['checkpoint'],'receipts':{},'logs':{}}
    for p in [*c.glob('*launched.json'),*r.glob('**/*identity*.json')]:
        d['receipts'][str(p)]=read(p)
        if 'pid' in d['receipts'][str(p)]:d['receipts'][str(p)]['current']=proc(d['receipts'][str(p)]['pid'])
    for p in [*r.glob('**/driver.log'),*r.glob('**/train.log'),*r.glob('**/worker*.log')]:
        if p.exists():d['logs'][str(p)]={'mtime':p.stat().st_mtime,'tail':p.read_text(errors='replace')[-8500:]}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or p.stat().st_uid!=20001:continue
        try:
            cmd=(p/'cmdline').read_bytes().decode(errors='replace')
            if str(r) in cmd:d.setdefault('processes',[]).append(proc(int(p.name)))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    out['runs'][key]=d
out['gpu']=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=20).stdout
print(json.dumps(out))
