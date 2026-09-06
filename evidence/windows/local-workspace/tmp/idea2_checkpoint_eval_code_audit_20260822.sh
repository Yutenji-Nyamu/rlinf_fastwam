#!/usr/bin/env bash
set -u

SRC=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight

printf '%s\n' '=== RUNNER 1-240 ==='
sed -n '1,240p' "$SRC/rlinf/runners/embodied_runner.py"
printf '%s\n' '=== RUNNER 280-380 ==='
sed -n '280,380p' "$SRC/rlinf/runners/embodied_runner.py"
printf '%s\n' '=== RUNNER 480-680 ==='
sed -n '480,680p' "$SRC/rlinf/runners/embodied_runner.py"
printf '%s\n' '=== ACTOR SAVE LOAD ==='
grep -nE '^    def (save|load)_checkpoint|full_weights|save_full_model_weights' "$SRC/rlinf/workers/actor/fsdp_actor_worker.py" || true
sed -n '1180,1245p' "$SRC/rlinf/workers/actor/fsdp_actor_worker.py"
printf '%s\n' '=== SFT ==='
for p in \
  /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle \
  /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50; do
  if [ -e "$p" ]; then
    printf 'FOUND\t%s\t' "$p"
    if [ -L "$p" ]; then readlink -f "$p"; else echo regular; fi
    du -shL "$p" 2>/dev/null || true
    find -L "$p" -maxdepth 3 -type f -printf '%P\t%s\n' | sort | head -n 30
  fi
done
printf '%s\n' '=== EVAL SEEDS ==='
python - <<'PY'
import json
from pathlib import Path
p=Path('/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/rlinf/envs/robotwin/seeds/eval_seeds.json')
d=json.loads(p.read_text())
x=d['adjust_bottle']
print(type(x).__name__, x.keys() if isinstance(x,dict) else None)
for k,v in x.items():
    print(k, len(v), v[:10])
PY
