#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
ACTION="$ROOT/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1"
ST="$ROOT/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2"
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389

printf 'NOW=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"
printf 'UPTIME='; uptime

for item in "ACTION:$ACTION" "ST_LOCALSHARD:$ST"; do
  label=${item%%:*}; run=${item#*:}
  printf '\n=== %s ===\n' "$label"
  pid=$(cat "$run/runtime/wrapper.pid" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then alive=1; else alive=0; fi
  printf 'wrapper_pid=%s alive=%s exit=' "$pid" "$alive"
  cat "$run/runtime/exit_code.txt" 2>/dev/null || echo pending
  stat -c 'driver_bytes=%s driver_mtime=%y' "$run/runtime/driver.log"
  printf 'fatal_count='; grep -aEic 'Traceback|CUDA out of memory|OutOfMemory|Error executing job|RayActorError|WorkerCrashedError|NCCL.*(error|failed)|non.?finite|ErrorInitializationFailed' "$run/runtime/driver.log" || true
  printf 'latest_complete='; grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$run/runtime/driver.log" | tail -n1 || echo none
  printf 'latest_progress='; grep -aoE 'Global Step:[[:space:]]+[0-9]+/100|Rollout Epoch:[[:space:]]+[0-9]+/4|Generating Rollout Epochs:[[:space:]]+[0-9]+/4' "$run/runtime/driver.log" | tail -n4 | tr '\n' ';'; echo
  printf 'latest_metric_lines:\n'
  grep -aE 'train/.*success|success_rate|actor/grad_norm|actor/dvac_weight_(mean|ess_fraction)|eval/.*success|fixed.*success' "$run/runtime/driver.log" | tail -n8 || true
  printf 'checkpoint_dirs='; find "$run" -type d -name 'global_step_*' | wc -l
  latest_ckpt=$(find "$run" -type d -name 'global_step_*' -printf '%f %p\n' | sort -V | tail -n1 | cut -d' ' -f2-)
  if [[ -n "$latest_ckpt" ]]; then
    printf 'latest_checkpoint=%s files=%s bytes=%s rank_local=%s metadata=%s\n' \
      "$latest_ckpt" "$(find "$latest_ckpt" -type f | wc -l)" "$(du -sb "$latest_ckpt" | awk '{print $1}')" \
      "$(find "$latest_ckpt" -type f -name 'checkpoint_rank_*.pt' | wc -l)" "$(find "$latest_ckpt" -type f -name '.metadata' | wc -l)"
  fi
  if [[ -s "$run/runtime/resource.csv" ]]; then printf 'resource_last='; tail -n1 "$run/runtime/resource.csv"; fi
done

printf '\n=== RAY ===\n'
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ['RAY_ADDRESS'], namespace='codex_server_readonly_20260830', logging_level='ERROR')
rows=ray.util.list_named_actors(all_namespaces=True)
for ns in ('RLinf','RLinf_1'):
    selected=[x for x in rows if x.get('namespace') == ns]
    print(f'{ns}_named_actors={len(selected)}')
ray.shutdown()
PY

printf '\n=== GPU ===\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
for dev in 0 1 2 3 4 5 6 7; do
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    user=$(ps -o user= -p "$pid" | xargs)
    used=$(nvidia-smi -i "$dev" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, -v p="$pid" '$1+0==p {gsub(/ /,"",$2); print $2; exit}')
    job=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    cmd=$(ps -o comm= -p "$pid" | xargs)
    printf 'gpu=%s pid=%s user=%s used_mib=%s job=%s cmd=%s\n' "$dev" "$pid" "$user" "${used:-?}" "${job:-none}" "$cmd"
  done < <(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
done

printf '\n=== HOST ===\n'
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
df -hT / /home /data | awk 'NR==1 || !seen[$7]++'
df -ih / /home /data | awk 'NR==1 || !seen[$6]++'
printf 'failed_units='; systemctl --failed --no-legend 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l
printf 'mihomo='; systemctl is-active mihomo 2>/dev/null || true
printf 'proxy_listener='; ss -lnt 2>/dev/null | awk '$4 ~ /127\.0\.0\.1:7890$/ {n++} END {print n+0}'
for target in https://github.com/ https://huggingface.co/; do
  printf 'proxy_%s=' "$(sed -E 's#https://([^/]+)/#\1#' <<< "$target")"
  curl -x http://127.0.0.1:7890 -I -L --max-time 8 -o /dev/null -sS -w 'http=%{http_code} time=%{time_total}s\n' "$target" || echo failed
done

printf '\n=== USER_RESOURCE_SUMMARY ===\n'
ps -eo user=,rss=,pcpu= --no-headers | awk '{rss[$1]+=$2; cpu[$1]+=$3; n[$1]++} END {for (u in n) if (rss[u] >= 1048576 || u ~ /^(chenyiteng|guorenjie|liwenbo|qiufuwen|yanchuhan|zhangwei|xiongzizhen|zhuanghuiping)$/) printf "%s processes=%d rss_gib=%.2f cpu_pct=%.1f\n",u,n[u],rss[u]/1048576,cpu[u]}' | sort
printf 'other_user_top_processes:\n'
ps -eo user=,pid=,etimes=,pcpu=,rss=,comm= --sort=-rss | awk '$1 !~ /^chenyit/ && $1 != "root" && $1 != "ray" && $1 != "nobody" && $5 >= 262144 {printf "%s pid=%s age_s=%s cpu=%s rss_mib=%.1f comm=%s\n",$1,$2,$3,$4,$5/1024,$6}' | head -n20
