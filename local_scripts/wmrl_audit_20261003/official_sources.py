"""Read pinned public source objects and save a bounded audit packet; no imports."""
import hashlib,json,subprocess,urllib.request,concurrent.futures
from pathlib import Path
OUT=Path(r'E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-audit-20261003/official-sources')
OUT.mkdir(parents=True,exist_ok=True)
ROOT=Path(r'C:/Users/86136/Documents/rl/.research-rlinf')
REV='d34d4c320d08cb982de034aa9a011f08dc0fa217'
FILES=['rlinf/algorithms/utils.py','rlinf/algorithms/losses.py','rlinf/algorithms/advantages.py','rlinf/envs/sim/world_model/env.py','rlinf/envs/sim/world_model/backend/wan.py','rlinf/workers/actor/fsdp_actor_worker.py','rlinf/workers/actor/embodied_fsdp_actor_worker.py','rlinf/utils/utils.py','rlinf/models/embodiment/openpi/tasks/rl.py','examples/embodiment/config/libero_goal_grpo_openpi_pi05.yaml','examples/embodiment/config/wan_libero_goal_grpo_openvlaoft.yaml','examples/embodiment/config/env/wan_libero_goal.yaml','examples/embodiment/config/model/pi0_5.yaml']
records=[]
for name in FILES:
    p=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','-C',str(ROOT),'show',f'{REV}:{name}'],capture_output=True)
    if p.returncode:
        records.append({'path':name,'error':p.stderr.decode()});continue
    dest=OUT/(name.replace('/','__'))
    dest.write_bytes(p.stdout)
    records.append({'path':name,'url':f'https://github.com/RLinf/RLinf/blob/{REV}/{name}','sha256':hashlib.sha256(p.stdout).hexdigest(),'local':str(dest)})
(OUT/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf8')
print(json.dumps(records,indent=2))
