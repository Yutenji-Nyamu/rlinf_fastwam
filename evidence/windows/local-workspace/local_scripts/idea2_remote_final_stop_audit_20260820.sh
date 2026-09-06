set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
config=/root/autodl-tmp/idea2_dvac_run_configs/idea2_dvac_sft_smoke_2gpu_2env_v1.yaml
output="$target/outputs/idea2_dvac_sft_smoke_2gpu_2env_v1"

printf 'SERVER_TIME=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %:z')"
printf 'HEAD=%s\n' "$(git -C "$target" rev-parse HEAD)"
printf 'REMOTE_HEAD=%s\n' \
  "$(git -C "$target" rev-parse refs/remotes/personal/codex/idea2-dvac-pi0-robotwin)"
printf 'DIRTY_COUNT=%s\n' "$(git -C "$target" status --short | wc -l)"
printf 'CONFIG_HASH=%s\n' "$(sha256sum "$config" | awk '{print $1}')"
if test -e "$output"; then
  printf 'OUTPUT_STATE=EXISTS\n'
  exit 1
else
  printf 'OUTPUT_STATE=ABSENT\n'
fi
printf 'GPU_COMPUTE_PROCESS_COUNT=%s\n' \
  "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)"
printf '%s\n' 'MATCHING_EVAL_RAY_PROCESSES_BEGIN'
ps -eo pid=,comm=,args= | awk -v self="$$" \
  '$1 != self && $2 != "awk" && /eval_embodied_agent[.]py|[r]aylet|ray::/ {print}'
printf '%s\n' 'MATCHING_EVAL_RAY_PROCESSES_END'
printf 'MEMORY_CURRENT=%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'MEMORY_EVENTS=%s\n' \
  "$(tr '\n' ';' < /sys/fs/cgroup/memory.events)"
test "$(git -C "$target" rev-parse HEAD)" = \
  61996e15cc7f5a32bd6012b61b20893d94636c82
test -z "$(git -C "$target" status --short)"
test "$(sha256sum "$config" | awk '{print $1}')" = \
  62207051678d2ad515c74aa574282b9bc04592f3b97ea10258d43263483b4f71
printf 'FINAL_STOP_AUDIT_PASS=1\n'
