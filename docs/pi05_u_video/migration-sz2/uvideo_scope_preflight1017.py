import json,subprocess,os,time
from pathlib import Path
D=Path('/data/chenyiteng/deployment-20261017/pi05-u-video-sz2-v2');O=Path('/data/chenyiteng/results/pi05-u-video/20261017-sz2-v2')
p=D/'graphics-scope.json';assert p.is_file() and p.stat().st_uid==os.getuid();p.chmod(0o644)
rows=[]
for g in [4,5]:
 env=dict(os.environ,**json.loads((O/f'environment-g{g}.json').read_text()))
 r=subprocess.run(['/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python','-B','-c','print("SCOPE_BOOTSTRAP_OK")'],env=env,capture_output=True,text=True,timeout=30)
 rows.append({'gpu':g,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0,repr(rows[-1])
(D/'bootstrap-checked.json').write_text(json.dumps({'time':time.time(),'rows':rows}));print(json.dumps(rows))
