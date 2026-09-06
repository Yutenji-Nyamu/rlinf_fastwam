#!/usr/bin/env bash
set -euo pipefail

RLT=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix
GRPO=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys2367-v2

printf 'MARKER=SZ_TWO_TRAINING_BRIEF_STATUS_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds

for item in "RLT:$RLT:250" "GRPO_DVAC:$GRPO:100"; do
  IFS=: read -r name run total <<<"$item"
  printf '=== %s ===\n' "$name"
  pid=$(cat "$run/runtime/wrapper.pid" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then echo wrapper=alive; else echo wrapper=dead; fi
  if [[ -s "$run/runtime/exit_code.txt" ]]; then printf 'exit_code='; cat "$run/runtime/exit_code.txt"; else echo exit_code=pending; fi
  python3 - "$run/runtime/driver.log" "$total" "$name" <<'PY'
import re, sys
from pathlib import Path

path, total, name = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
text = path.read_text(encoding='utf-8', errors='replace')
text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text).replace('\r', '\n')
matches = list(re.finditer(rf'Global Step:\s*(\d+)/{total}', text))
if not matches:
    print('latest_complete_step=none')
    raise SystemExit
m = matches[-1]
step = int(m.group(1))
segment = text[m.start():]
print(f'latest_complete_step={step}/{total}')

def first(key):
    hit = re.search(rf'(?<![\w/]){re.escape(key)}=([-+0-9.eE]+)', segment)
    return hit.group(1) if hit else None

success = re.findall(r'(?<![\w/])success_once=([-+0-9.eE]+)', segment)
if success:
    print('train_success=' + success[0])
if len(success) > 1:
    print('fixed_eval_success=' + success[1])
keys = (
    ('update_step', 'rlt/update_step'),
    ('actor_loss', 'sac/actor_loss'),
    ('critic_loss', 'sac/critic_loss'),
    ('approx_kl', 'policy/approx_kl'),
    ('clip_fraction', 'policy/clip_fraction'),
    ('grad_norm', 'actor/grad_norm'),
    ('dvac_ess', 'dvac/weight_ess'),
)
for out, key in keys:
    value = first(key)
    if value is not None:
        print(f'{out}={value}')

fatal_pattern = re.compile(r'Traceback|OutOfMemory|WorkerCrashed|nonfinite', re.I)
print('fatal_matches=' + str(len(fatal_pattern.findall(text))))
PY
  printf '%s\n' 'checkpoints:'
  find "$run" -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -Vu | tail -n 5 || true
done

printf '%s\n' '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
grep -E '^(MemAvailable|SwapFree):' /proc/meminfo
printf 'MARKER=SZ_TWO_TRAINING_BRIEF_STATUS_OK\n'
