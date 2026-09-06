#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
GRPO=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
RLT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
DSRL=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

printf '=== IDENTITY_TIME ===\n'
date --iso-8601=seconds
hostname
id

printf '=== GRPO_EXIT_TIMELINE ===\n'
for file in launch_manifest.txt driver.pid resource_observer.pid driver.exit resource_observer.exit ray_log_snapshot.txt; do
  printf '%s\n' "-- $file"
  if test -f "$RUN/$file"; then
    stat -c 'mtime=%y size=%s inode=%i' "$RUN/$file"
    cat "$RUN/$file"
  else
    printf 'missing\n'
  fi
done
printf '%s\n' '-- primary log stats and tails'
for file in driver.log metrics.log resource.csv resource_observer.log; do
  if test -f "$RUN/$file"; then stat -c '%n mtime=%y size=%s' "$RUN/$file"; else printf '%s missing\n' "$file"; fi
done
printf '%s\n' '-- last 120 driver lines'
tail -n 120 "$RUN/driver.log" 2>/dev/null || true
printf '%s\n' '-- last 40 resource rows'
tail -n 40 "$RUN/resource.csv" 2>/dev/null || true
printf '%s\n' '-- final Ray logs inventory'
find "$RUN/ray_logs_final" -maxdepth 1 -type f -printf '%f %s %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort || true
printf '%s\n' '-- targeted final Ray diagnostics'
find "$RUN/ray_logs_final" -maxdepth 1 -type f 2>/dev/null \
  -exec grep -aHniE 'error|fatal|failed|disconnect|heartbeat|timeout|SIG|kill|oom|exception|shutdown|stopping|stopped' {} + \
  | tail -n 240 || true

printf '=== RAY_AND_PORT_ISOLATION ===\n'
printf '%s\n' '-- current owned Ray processes'
ps -u "$(id -u)" -o pid=,ppid=,lstart=,etime=,rss=,stat=,comm=,args= \
  | grep -E 'raylet|gcs_server|ray::|train_embodied_agent' | grep -v grep || true
printf '%s\n' '-- Ray session directories'
find /tmp/ray -mindepth 1 -maxdepth 1 -type d -name 'session_*' \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort || true
printf 'session_latest='; readlink -f /tmp/ray/session_latest 2>/dev/null || true
printf '%s\n' '-- relevant listeners'
ss -lntp 2>/dev/null | grep -E ':(6379|6380|8265|10001|7890|9090)[[:space:]]' || true

printf '=== WORKTREE_AND_RUNTIME_SEPARATION ===\n'
for item in "GRPO:$GRPO" "RLT:$RLT" "DSRL:$DSRL"; do
  name=${item%%:*}; path=${item#*:}
  printf '%s path=%s\n' "$name" "$path"
  if git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'head='; git -C "$path" rev-parse HEAD
    printf 'branch='; git -C "$path" branch --show-current
    printf 'commit_time='; git -C "$path" show -s --format='%cI' HEAD
    printf 'status='; test -z "$(git -C "$path" status --porcelain=v1)" && echo clean || { echo dirty; git -C "$path" status --short; }
  else
    printf 'missing\n'
  fi
done
printf '%s\n' '-- tracked GRPO files modified during 19:00-20:10 CST window'
find "$GRPO" -xdev -type f -newermt '2026-08-23 19:00:00 +0800' ! -newermt '2026-08-23 20:10:00 +0800' \
  -not -path '*/.git/*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | head -n 120 || true
printf '%s\n' '-- RLT/DSRL files modified in same window'
for path in "$RLT" "$DSRL"; do
  printf 'path=%s\n' "$path"
  find "$path" -xdev -type f -newermt '2026-08-23 19:00:00 +0800' ! -newermt '2026-08-23 20:10:00 +0800' \
    -not -path '*/.git/*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | head -n 160 || true
done
printf '%s\n' '-- shared venv recent file count/sample in same window'
recent_count=$(find "$VENV" -xdev -type f -newermt '2026-08-23 19:00:00 +0800' ! -newermt '2026-08-23 20:10:00 +0800' 2>/dev/null | wc -l)
printf 'recent_venv_file_count=%s\n' "$recent_count"
find "$VENV" -xdev -type f -newermt '2026-08-23 19:00:00 +0800' ! -newermt '2026-08-23 20:10:00 +0800' \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | head -n 80 || true
stat -c 'venv_cfg mtime=%y size=%s' "$VENV/pyvenv.cfg" 2>/dev/null || true

printf '=== NETWORK_LIGHT ===\n'
printf 'mihomo_service='; systemctl is-active mihomo 2>/dev/null || true
printf 'proxy_profile_mtime='; stat -c '%y' /etc/profile.d/mihomo-proxy.sh 2>/dev/null || true
ss -ltn 2>/dev/null | grep -E '127\.0\.0\.1:(7890|9090)' || true
probe() {
  label=$1; shift
  printf '%s ' "$label"
  "$@" -o /dev/null -sS -L --connect-timeout 5 --max-time 12 \
    -w 'code=%{http_code} connect=%{time_connect} total=%{time_total} remote=%{remote_ip}\n' \
    || printf 'curl_exit=%s\n' "$?"
}
probe direct_github env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy curl https://github.com/
probe direct_hf env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy curl 'https://huggingface.co/api/models?limit=1'
probe proxy_github curl -x http://127.0.0.1:7890 https://github.com/
probe proxy_hf curl -x http://127.0.0.1:7890 'https://huggingface.co/api/models?limit=1'

printf 'SZ_PROXY_GRPO_USER_AUDIT_OK\n'
