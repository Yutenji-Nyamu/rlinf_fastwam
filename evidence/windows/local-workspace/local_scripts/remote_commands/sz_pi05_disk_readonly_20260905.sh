#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,json,subprocess
from pathlib import Path
run=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
tz=datetime.timezone(datetime.timedelta(hours=8))
d={'time':datetime.datetime.now(tz).isoformat(),'run':str(run)}
cmd=['timeout','35s','ionice','-c','3','nice','-n','19','du','-x','-B1','--max-depth=2',str(run)]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
d['du']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
d['checkpoints']=[{'path':str(p),'bytes':p.stat().st_size} for p in run.glob('*/checkpoints/global_step_*/actor/**/*') if p.is_file()]
d['df']=subprocess.run(['df','-B1','/data'],capture_output=True,text=True).stdout
print(json.dumps(d))
PY
