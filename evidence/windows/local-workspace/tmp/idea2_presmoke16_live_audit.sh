set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
branch=codex/idea2-dvac-pi0-robotwin
head=61996e15cc7f5a32bd6012b61b20893d94636c82
robotwin=/root/autodl-tmp/RoboTwin_RLinf
checkpoint=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
old_config=/root/autodl-tmp/idea2_dvac_run_configs/idea2_dvac_sft_smoke_2gpu_2env_v1.yaml
new_config=/root/autodl-tmp/idea2_dvac_run_configs/idea2_dvac_sft_smoke_2gpu_16env_v1.yaml
new_output="$target/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1"

printf 'SERVER_TIME=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %:z')"
printf 'HOST=%s\nPWD=%s\nUID=%s\n' "$(hostname)" "$PWD" "$(id -u)"
printf 'SOURCE_HEAD=%s\nREMOTE_HEAD=%s\nBRANCH=%s\nDIRTY_COUNT=%s\n' \
  "$(git -C "$target" rev-parse HEAD)" \
  "$(git -C "$target" rev-parse personal/$branch)" \
  "$(git -C "$target" branch --show-current)" \
  "$(git -C "$target" status --short | wc -l)"
printf 'ROBOTWIN_HEAD=%s\n' "$(git -C "$robotwin" rev-parse HEAD)"
printf 'CHECKPOINT_BYTES=%s\n' "$(du -sb "$checkpoint" | cut -f1)"
printf 'OLD_CONFIG_SHA256=%s\n' "$(sha256sum "$old_config" | cut -d' ' -f1)"
if test -e "$new_config"; then printf 'NEW_CONFIG_STATE=EXISTS\n'; else printf 'NEW_CONFIG_STATE=ABSENT\n'; fi
if test -e "$new_output"; then printf 'NEW_OUTPUT_STATE=EXISTS\n'; else printf 'NEW_OUTPUT_STATE=ABSENT\n'; fi
printf '%s\n' 'GPU_SNAPSHOT_BEGIN'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits
printf '%s\n' 'GPU_SNAPSHOT_END'
printf 'GPU_COMPUTE_PROCESS_COUNT=%s\n' \
  "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)"
printf '%s\n' 'MATCHING_EVAL_RAY_PROCESSES_BEGIN'
ps -eo pid=,comm=,args= | awk -v self="$$" \
  '$1 != self && $2 != "awk" && /eval_embodied_agent[.]py|[r]aylet|ray::/ {print}'
printf '%s\n' 'MATCHING_EVAL_RAY_PROCESSES_END'
printf 'MEMORY_CURRENT=%s\nMEMORY_HIGH=%s\nMEMORY_MAX=%s\nMEMORY_EVENTS=%s\n' \
  "$(cat /sys/fs/cgroup/memory.current)" \
  "$(cat /sys/fs/cgroup/memory.high)" \
  "$(cat /sys/fs/cgroup/memory.max)" \
  "$(tr '\n' ';' < /sys/fs/cgroup/memory.events)"
df -h /root/autodl-tmp /dev/shm

test "$(git -C "$target" rev-parse HEAD)" = "$head"
test "$(git -C "$target" rev-parse personal/$branch)" = "$head"
test "$(git -C "$target" branch --show-current)" = "$branch"
test -z "$(git -C "$target" status --porcelain)"
test "$(git -C "$robotwin" rev-parse HEAD)" = 481380fbd97cbf9ff830aedfb2279851e1e58969
test -d "$checkpoint"
test -f "$old_config"
test ! -e "$new_config"
test ! -e "$new_output"
printf 'PRESMOKE16_LIVE_AUDIT_PASS=1\n'

