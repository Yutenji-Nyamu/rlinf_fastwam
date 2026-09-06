#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

date --iso-8601=seconds
printf 'head='; git -C "$WT" rev-parse HEAD
printf 'status='; git -C "$WT" status --porcelain | wc -l

echo '=== seed banks ==='
python3 - "$WT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for rel in ('rlinf/envs/robotwin/seeds/train_seeds.json', 'rlinf/envs/robotwin/seeds/eval_seeds.json'):
    p = root / rel
    data = json.loads(p.read_text())
    value = data.get('move_pillbottle_pad')
    print(rel, 'present=', 'move_pillbottle_pad' in data, 'type=', type(value).__name__, 'count=', None if value is None else len(value))
    print('value=', value)
PY

echo '=== reward/task references in current RLinf ==='
grep -RIn --exclude-dir=.git --exclude='*.pyc' 'move_pillbottle_pad' \
  "$WT/rlinf/envs/robotwin" "$WT/examples/embodiment/config" 2>/dev/null || true

echo '=== current Sidney task config ==='
sed -n '1,260p' "$WT/examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml"

echo '=== base env task configs ==='
find "$WT/examples/embodiment/config" -maxdepth 2 -type f \
  \( -iname '*move_pillbottle_pad*' -o -iname '*move_stapler_pad*' \) -print | sort

echo '=== RoboTwin task references/config ==='
find "$ROBOTWIN" -path '*/.git' -prune -o -type f \
  \( -iname '*move_pillbottle_pad*' -o -iname '*move_stapler_pad*' \) -print 2>/dev/null | head -80
grep -RIn --exclude-dir=.git --exclude='*.pyc' 'move_pillbottle_pad' \
  "$ROBOTWIN/task_config" "$ROBOTWIN/env_cfg" "$ROBOTWIN/script" 2>/dev/null | head -120 || true

echo '=== official step limit entries ==='
find "$ROBOTWIN" -type f -name '_eval_step_limit.yml' -print -exec grep -nE 'move_(pillbottle|stapler)_pad' {} \; 2>/dev/null || true

echo '=== task class metadata ==='
python3 - "$ROBOTWIN" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1])
for task in ('move_pillbottle_pad','move_stapler_pad'):
    hits=[]
    for p in root.rglob('*.py'):
        if '.git' in p.parts:
            continue
        try:
            s=p.read_text(errors='ignore')
        except OSError:
            continue
        if task in s:
            hits.append(str(p.relative_to(root)))
    print(task, hits[:30])
PY

echo '=== task instruction ==='
cat "$ROBOTWIN/description/task_instruction/move_pillbottle_pad.json"

echo '=== task implementation ==='
sed -n '1,280p' "$ROBOTWIN/envs/move_pillbottle_pad.py"

echo '=== current adapter reward path ==='
grep -RIn --exclude-dir=.git --exclude='*.pyc' -E 'reward|success|eval_success|take_action' \
  "$WT/rlinf/envs/robotwin" 2>/dev/null | head -240 || true
