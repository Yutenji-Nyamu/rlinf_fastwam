#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

python3 - "$WT" "$ROBOTWIN" <<'PY'
import ast, json, sys
from pathlib import Path

wt, rt = map(Path, sys.argv[1:])
for split in ('train', 'eval'):
    d=json.loads((wt/f'rlinf/envs/robotwin/seeds/{split}_seeds.json').read_text())
    for task in ('move_stapler_pad','move_pillbottle_pad'):
        v=d[task]
        s=v['success_seeds']
        print(f'{split}.{task}: task_name={v["task_name"]} count={len(s)} first={s[:4]} last={s[-4:]}')

for task in ('move_stapler_pad','move_pillbottle_pad'):
    p=rt/f'description/task_instruction/{task}.json'
    d=json.loads(p.read_text())
    print(f'prompt.{task}: full={d.get("full_description")!r} seen={len(d.get("seen",[]))} unseen={len(d.get("unseen",[]))}')
    ep=rt/f'envs/{task}.py'
    tree=ast.parse(ep.read_text())
    classes=[n.name for n in tree.body if isinstance(n,ast.ClassDef)]
    funcs=[]
    for n in tree.body:
        if isinstance(n,ast.ClassDef):
            funcs.extend(x.name for x in n.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)))
    print(f'impl.{task}: file={ep.name} classes={classes} check_success={"check_success" in funcs}')

limit={}
for line in (rt/'task_config/_eval_step_limit.yml').read_text().splitlines():
    if ':' in line and not line.lstrip().startswith('#'):
        k,v=line.split(':',1)
        if k.strip() in ('move_stapler_pad','move_pillbottle_pad'):
            limit[k.strip()]=v.strip()
print('limits=',limit)

cfg=(wt/'examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml').read_text()
print('config_task_occurrences=',cfg.count('move_stapler_pad'))
for i,line in enumerate(cfg.splitlines(),1):
    if 'move_stapler_pad' in line:
        print(f'cfg:{i}:{line.strip()}')
PY

sed -n '405,500p' "$WT/rlinf/envs/robotwin/robotwin_env.py"
sed -n '1,120p' "$ROBOTWIN/envs/move_pillbottle_pad.py"
sed -n '120,260p' "$ROBOTWIN/envs/move_pillbottle_pad.py"
