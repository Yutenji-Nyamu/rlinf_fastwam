#!/usr/bin/env bash
set -u

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4
CKPT="$RUN/robotwin_grpo_openpi_dvac_global_z_matched/checkpoints/global_step_30"
LOG="$RUN/runtime/driver.log"

echo '=== identity_and_source ==='
TZ=Asia/Shanghai date --iso-8601=seconds
id
printf 'head='; git -C "$WT" rev-parse HEAD
printf 'dirty='; git -C "$WT" status --short | wc -l

echo '=== exact_exit_tail ==='
printf 'exit_code='; cat "$RUN/runtime/exit_code.txt"
tail -n 100 "$LOG" | grep -aE 'Global Step:|Generating Rollout Epochs:|Traceback|WorkerCrashed|SIGTERM|SIGKILL|killed|ActorDied|RayActorError|ConnectionError|EOFError|Exception|Error' || true

echo '=== checkpoint30_contract ==='
test -d "$CKPT"
printf 'files='; find "$CKPT" -type f | wc -l
printf 'sidecars='; find "$CKPT" -type f -name 'dvac_state_rank*.json' | wc -l
find "$CKPT" -maxdepth 3 -type f -printf '%P %s\n' | sort
while IFS= read -r f; do
  "$VENV/bin/python" - "$f" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
print(p.name, sorted(d), 'mode=', d.get('mode'), 'selected_l=', d.get('selected_l'), 'world_size=', d.get('world_size'))
PY
done < <(find "$CKPT" -type f -name 'dvac_state_rank*.json' | sort)

echo '=== gpu4567_and_owned_processes ==='
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 4,5,6,7 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
ps -u chenyiteng -o pid=,ppid=,etimes=,rss=,stat=,comm=,args= --sort=-rss | grep -E 'ray::|train_embodied_agent|RolloutGroup|EnvWorker|ActorWorker|wrapper.sh' | head -n 80 || true

echo '=== shared_ray ==='
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" list actors --filter 'state=ALIVE' --format=table 2>/dev/null || true

echo '=== resume_api ==='
grep -R -n -E 'resume_dir|load_checkpoint|dvac_state_rank|load_state_dict' \
  --include='*.py' --include='*.yaml' \
  "$WT/rlinf" "$WT/examples/embodiment" \
  | grep -E 'resume_dir|checkpoint|dvac' | head -n 180

echo SZ_DVAC_GRPO_V4_RESUME_PREFLIGHT_OK
