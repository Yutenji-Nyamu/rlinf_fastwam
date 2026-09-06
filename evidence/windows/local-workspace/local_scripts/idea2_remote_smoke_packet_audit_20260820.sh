set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
robotwin=/root/autodl-tmp/RoboTwin_RLinf
checkpoint=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
seeds="$target/rlinf/envs/robotwin/seeds/eval_seeds.json"
candidate_output="$target/outputs/idea2_dvac_sft_smoke_2gpu_2env_v1"
runtime=/root/autodl-tmp/RLinf/.venv/bin/python

printf 'SERVER_TIME=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %:z')"
printf 'HOST=%s\n' "$(hostname)"
printf 'SOURCE_HEAD=%s\n' "$(git -C "$target" rev-parse HEAD)"
printf 'SOURCE_BRANCH=%s\n' "$(git -C "$target" branch --show-current)"
printf 'SOURCE_DIRTY_COUNT=%s\n' "$(git -C "$target" status --short | wc -l)"
printf 'REMOTE_TRACKING_HEAD=%s\n' \
  "$(git -C "$target" rev-parse refs/remotes/personal/codex/idea2-dvac-pi0-robotwin)"
printf 'ROBOTWIN_HEAD=%s\n' "$(git -C "$robotwin" rev-parse HEAD)"
printf 'RUNTIME=%s\n' "$runtime"
test -x "$runtime"

printf 'CHECKPOINT=%s\n' "$checkpoint"
test -d "$checkpoint"
du -sh "$checkpoint"
norm_stats="$checkpoint/physical-intelligence/robotwin/norm_stats.json"
test -f "$norm_stats"
sha256sum "$norm_stats"

printf 'SEEDS=%s\n' "$seeds"
test -f "$seeds"
sha256sum "$seeds"
"$runtime" - "$seeds" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
values = payload["adjust_bottle"]["success_seeds"]
print(f"ADJUST_BOTTLE_SEED_COUNT={len(values)}")
print(f"ADJUST_BOTTLE_SEEDS_UNIQUE={int(len(values) == len(set(values)))}")
PY

for assets in \
  /root/autodl-tmp/RoboTwin_RLinf/assets \
  /root/autodl-tmp/RoboTwin/assets
do
  if test -d "$assets"; then
    printf 'ASSETS_CANDIDATE=%s\n' "$assets"
    printf 'ASSETS_RESOLVED=%s\n' "$(readlink -f "$assets")"
  fi
done

if test -e "$candidate_output"; then
  printf 'CANDIDATE_OUTPUT_STATE=EXISTS\n'
  exit 1
else
  printf 'CANDIDATE_OUTPUT_STATE=ABSENT\n'
fi

printf '%s\n' '--- GPU SNAPSHOT ---'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits
printf '%s\n' '--- CGROUP SNAPSHOT ---'
for field in memory.current memory.high memory.max memory.peak memory.events
do
  if test -r "/sys/fs/cgroup/$field"; then
    printf '%s=' "$field"
    tr '\n' ';' < "/sys/fs/cgroup/$field"
    printf '\n'
  fi
done
printf '%s\n' '--- DISK/SHM ---'
df -h /root/autodl-tmp /dev/shm
printf 'SMOKE_PACKET_AUDIT_PASS=1\n'
