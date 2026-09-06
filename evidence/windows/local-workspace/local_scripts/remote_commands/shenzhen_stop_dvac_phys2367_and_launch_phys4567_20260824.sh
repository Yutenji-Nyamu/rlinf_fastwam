#!/usr/bin/env bash
set -euo pipefail

OLD_NAME=dvac-global-z-formal100-4gpu32x8-g8-phys2367-v2
NEW_NAME=dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3
BASE=/data/chenyiteng/results/rlinf-shenzhen/grpo
OLD_RUN=$BASE/runs/$OLD_NAME
OLD_PACKET=$BASE/packets/$OLD_NAME
NEW_RUN=$BASE/runs/$NEW_NAME
NEW_PACKET=$BASE/packets/$NEW_NAME

printf 'MARKER=SZ_STOP_DVAC_PHYS2367_LAUNCH_PHYS4567_V1\n'
test -d "$OLD_RUN/runtime"
test -d "$OLD_PACKET"
test ! -e "$NEW_RUN"
test ! -e "$NEW_PACKET"

pid=$(cat "$OLD_RUN/runtime/wrapper.pid")
pgid=$(cat "$OLD_RUN/runtime/owned.pgid")
test "$pid" = "$pgid"
test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
ps -o args= -p "$pid" | grep -F "$OLD_RUN/runtime/wrapper.sh" >/dev/null
printf 'old_wrapper_pid=%s old_pgid=%s\n' "$pid" "$pgid"

kill -TERM -- "-$pgid"
for _ in $(seq 1 90); do
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 2
done
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL -- "-$pgid"
  sleep 3
fi
if kill -0 "$pid" 2>/dev/null; then
  echo 'old owned process group did not exit' >&2
  exit 20
fi

observer=$(cat "$OLD_RUN/runtime/observer.pid" 2>/dev/null || true)
if [[ -n "$observer" ]] && kill -0 "$observer" 2>/dev/null; then
  kill -TERM -- "-$observer" 2>/dev/null || kill -TERM "$observer" 2>/dev/null || true
fi

for _ in $(seq 1 60); do
  if ! nvidia-smi -i 2,3,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then break; fi
  sleep 2
done
if nvidia-smi -i 2,3,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'old GRPO GPU processes remain; refusing new launch' >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  exit 21
fi

mkdir -p "$NEW_PACKET"
cp -a "$OLD_PACKET/." "$NEW_PACKET/"
python3 - "$NEW_PACKET" "$OLD_NAME" "$NEW_NAME" <<'PY'
from pathlib import Path
import sys

root, old_name, new_name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for name in ('resolved.yaml', 'contract.json', 'command.txt', 'launch.sh'):
    path = root / name
    text = path.read_text(encoding='utf-8')
    text = text.replace(old_name, new_name)
    text = text.replace('2,3,6,7', '4,5,6,7')
    text = text.replace('[2, 3, 6, 7]', '[4, 5, 6, 7]')
    text = text.replace(
        'gpu2_used_mib,gpu2_util_pct,gpu3_used_mib,gpu3_util_pct',
        'gpu4_used_mib,gpu4_util_pct,gpu5_used_mib,gpu5_util_pct',
    )
    text = text.replace('PHYS2367', 'PHYS4567')
    path.write_text(text, encoding='utf-8')
PY
chmod 700 "$NEW_PACKET/launch.sh"
bash -n "$NEW_PACKET/launch.sh"
if grep -R -E 'phys2367-v2|2,3,6,7|gpu2_used_mib|gpu3_used_mib' "$NEW_PACKET"; then
  echo 'stale old placement remains in new packet' >&2
  exit 22
fi
python3 - "$NEW_PACKET/contract.json" <<'PY'
import json, sys
from pathlib import Path
c = json.loads(Path(sys.argv[1]).read_text())
assert c['physical_gpus'] == [4, 5, 6, 7]
assert c['outer_steps'] == 100
assert c['train_envs'] == 32 and c['rollout_epochs'] == 8
assert c['trajectories_per_step'] == 256
assert c['global_batch'] == 512 and c['micro_batch'] == 32
assert c['dvac']['selected_l'] == 3 and c['dvac']['weight_range'] == [0, 2]
PY

printf 'old_exit='; cat "$OLD_RUN/runtime/exit_code.txt" 2>/dev/null || echo pending
printf '%s\n' '=== GPU BEFORE NEW LAUNCH ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
bash "$NEW_PACKET/launch.sh"
printf 'new_packet=%s\nnew_run=%s\n' "$NEW_PACKET" "$NEW_RUN"
printf 'MARKER=SZ_STOP_DVAC_PHYS2367_LAUNCH_PHYS4567_OK\n'
