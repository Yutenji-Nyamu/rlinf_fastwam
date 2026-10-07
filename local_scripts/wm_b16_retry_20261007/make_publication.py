"""Publish reviewed B16 source, current note and small receipts only."""
import ast,base64,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;W=H.parents[1];D=W/'docs/world-model/task_reward_plan_20261006'
sys.path.insert(0,str(H.parent/'lift_two_gpu_20261006'))
from make_result_publication import REMOTE
followup='--result' in sys.argv
prior=json.loads((D/('two_gpu_b16_published.json' if followup else 'rlt45_return_final_published.json')).read_text())
names=['docs/world-model/task_reward_plan_20261006/'+n for n in ['README.md','two_gpu_b16_restart_20261007.md','two_gpu_b16_light.json','two_gpu_b32_execution.md']]
names+=['local_scripts/wm_b16_retry_20261007/'+n for n in ['owner.py','prepare.py','build_owner.py','make_package.py','launch_remote.py','start_audit_remote.py','status_remote.py','record.py','make_publication.py']]
if followup:names.append('docs/world-model/task_reward_plan_20261006/two_gpu_b16_published.json')
files={}
for n in names:
 b=(W/n).read_bytes().replace(b'\r\n',b'\n');assert len(b)<262144
 if n.endswith('.py'):ast.parse(b)
 files[n]={'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
release='lift-b16-formal-started-20261007-v1' if followup else 'lift-b16-formal-20261007-v1'
P={'files':files,'prior':prior['commit'],'branch':prior['branch'],'runtime_head':prior['runtime_head'],'release_name':release}
remote=REMOTE
a=remote.index("assert P['release_name'] in ");b=remote.index('\n',a);remote=remote[:a]+"assert P['release_name']=="+repr(release)+remote[b:]
a=remote.index("F=S/'lift-two-gpu-from0-v2'");b=remote.index('assert not E.exists()',a)
check="""F=S/'lift-two-gpu-b16-20261007-v1'
for name in ['owner.py','prepare.py']:
 assert hashlib.sha256((F/'code'/name).read_bytes()).hexdigest()==P['files']['local_scripts/wm_b16_retry_20261007/'+name]['sha256']
"""
remote=remote[:a]+check+remote[b:]
a=remote.index("message='Record fresh");b=remote.index("git(W,'commit'",a)
remote=remote[:a]+"message='Run two-card lift WMRL with B16 and consolidate formal owner'\n"+remote[b:]
source='P = '+repr(P)+'\n'+remote;ast.parse(source)
(H/'publication_remote.py').write_text(source,encoding='utf-8',newline='\n')
print(json.dumps({'files':len(files),'prior':P['prior'],'release':release}))
