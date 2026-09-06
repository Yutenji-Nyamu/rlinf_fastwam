#!/usr/bin/env bash

set +e

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
DRIVER=1375834

section() {
  printf '\n===== %s =====\n' "$1"
}

section timestamp_identity
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
hostname
uname -a
id
groups
printf 'cwd=%s\n' "$PWD"
sudo -n -l 2>&1

section gpu_inventory
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
printf '%s\n' '-- compute applications --'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1
printf '%s\n' '-- compute pid owners --'
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nu); do
  test -r "/proc/$pid/status" || continue
  owner=$(stat -c '%U' "/proc/$pid" 2>/dev/null)
  name=$(awk '/^Name:/{print $2}' "/proc/$pid/status" 2>/dev/null)
  rss=$(awk '/^VmRSS:/{print $2" "$3}' "/proc/$pid/status" 2>/dev/null)
  printf 'pid=%s owner=%s name=%s rss=%s\n' "$pid" "$owner" "$name" "$rss"
done

section ppo_driver_and_progress
if kill -0 "$DRIVER" 2>/dev/null; then
  printf 'driver_alive=1 pid=%s\n' "$DRIVER"
  ps -o user,pid,ppid,stat,etimes,%cpu,%mem,rss,vsz,comm,args -p "$DRIVER"
else
  printf 'driver_alive=0 pid=%s\n' "$DRIVER"
fi
printf '%s\n' '-- owned RLinf processes --'
ps -eo user,pid,ppid,stat,%cpu,%mem,rss,etimes,comm --sort=-rss | awk '$10 ~ /^(ray::|raylet|gcs_server|python)/ {print}' | head -n 120
printf '%s\n' '-- recent progress lines --'
if test -r "$RUN/driver.log"; then
  grep -aE 'Global Step [0-9]+/100|Generating Rollout Epochs|eval/success_once|Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|SIGKILL|NCCL' "$RUN/driver.log" | tail -n 80
  printf 'fatal_scan_count='
  grep -aiEc 'Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|SIGKILL|NCCL.*(error|failed)' "$RUN/driver.log"
  stat -c 'driver_log_bytes=%s mtime=%y' "$RUN/driver.log"
else
  printf 'driver_log_missing=%s\n' "$RUN/driver.log"
fi
printf '%s\n' '-- checkpoint inventory --'
find "$RUN/robotwin_ppo_openpi/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V
du -sh "$RUN" "$RUN"/robotwin_ppo_openpi/checkpoints/global_step_* 2>/dev/null

section memory_host_and_cgroup
free -h
grep -E '^(MemTotal|MemAvailable|MemFree|Cached|Buffers|SwapTotal|SwapFree|Slab|SReclaimable|Shmem|AnonPages|Mapped|PageTables):' /proc/meminfo
vmstat 1 3
if test -r "/proc/$DRIVER/cgroup"; then
  printf '%s\n' '-- driver cgroup --'
  cat "/proc/$DRIVER/cgroup"
  cg_rel=$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER/cgroup")
  cg="/sys/fs/cgroup${cg_rel}"
  printf 'driver_cgroup_path=%s\n' "$cg"
  for f in memory.current memory.peak memory.high memory.max memory.swap.current memory.events; do
    test -r "$cg/$f" && { printf -- '-- %s --\n' "$f"; cat "$cg/$f"; }
  done
fi
printf '%s\n' '-- per-user process RSS aggregate --'
ps -eo user=,rss=,%cpu= | awk '{n[$1]++; rss[$1]+=$2; cpu[$1]+=$3} END {for (u in n) printf "%s processes=%d rss_kib=%d cpu_sum=%.1f\n",u,n[u],rss[u],cpu[u]}' | sort -k3,3nr
printf '%s\n' '-- top RSS processes --'
ps -eo user,pid,ppid,stat,%cpu,%mem,rss,vsz,etimes,comm --sort=-rss | head -n 35

section envworker_memory_breakdown
env_pids=$(ps -eo pid=,comm= | awk '$2 ~ /^ray::EnvWorker/ {print $1}' | sort -nu)
printf 'envworker_pids=%s\n' "$(printf '%s ' $env_pids)"
for pid in $env_pids; do
  test -r "/proc/$pid/status" || continue
  printf -- '--- pid=%s owner=%s fd_count=%s ---\n' "$pid" "$(stat -c '%U' "/proc/$pid")" "$(find "/proc/$pid/fd" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)"
  grep -E '^(Name|State|Threads|VmPeak|VmSize|VmRSS|RssAnon|RssFile|RssShmem|VmData|VmSwap|HugetlbPages):' "/proc/$pid/status"
  if test -r "/proc/$pid/smaps_rollup"; then
    grep -E '^(Rss|Pss|Pss_Anon|Pss_File|Pss_Shmem|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Referenced|Anonymous|LazyFree|AnonHugePages|Swap):' "/proc/$pid/smaps_rollup"
  fi
done

section storage_and_inodes
df -hT / /home /data
df -ih / /home /data
findmnt -T / -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS
findmnt -T /home -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS
findmnt -T /data -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS
du -sh /home/chenyiteng /data/chenyiteng /home/chenyiteng/venvs /home/chenyiteng/cache /data/chenyiteng/projects /data/chenyiteng/models /data/chenyiteng/results 2>/dev/null

section network_proxy_small_probes
ip -brief address
ip route
printf '%s\n' '-- resolver --'
sed -n '1,80p' /etc/resolv.conf
getent ahostsv4 github.com | head -n 3
systemctl is-active mihomo.service 2>&1
systemctl is-enabled mihomo.service 2>&1
ss -lnt | grep -E '127\.0\.0\.1:(7890|9090)' || true
source /etc/profile.d/mihomo-proxy.sh 2>/dev/null
env | grep -iE '^(http|https|all|no)_proxy=' | sed -E 's#(https?://)[^/@]+@#\1REDACTED@#I' | sort
probe() {
  label=$1
  url=$2
  curl -sSIL --max-time 15 --connect-timeout 8 -o /dev/null -w "$label http=%{http_code} remote=%{remote_ip} connect=%{time_connect} total=%{time_total} bytes=%{size_download}\n" "$url" 2>&1
}
probe github https://github.com/
probe pypi https://pypi.org/simple/pip/
probe huggingface https://huggingface.co/api/models/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
probe modelscope https://www.modelscope.cn/api/v1/models/iic/CV_crowd-counting_dmap_estimation
printf '%s\n' '-- direct-vs-proxy github --'
curl --noproxy '*' -sSIL --max-time 10 --connect-timeout 5 -o /dev/null -w 'github_direct http=%{http_code} remote=%{remote_ip} connect=%{time_connect} total=%{time_total}\n' https://github.com/ 2>&1
curl --proxy http://127.0.0.1:7890 -sSIL --max-time 10 --connect-timeout 5 -o /dev/null -w 'github_proxy http=%{http_code} remote=%{remote_ip} connect=%{time_connect} total=%{time_total}\n' https://github.com/ 2>&1

section fastwam_breakpoint
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
FENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
FCACHE=/home/chenyiteng/cache/fastwam-7faa
for target in "$FW" "$FENV" "$FCACHE" /data/chenyiteng/models/fastwam /data/chenyiteng/results/fastwam-standalone; do
  if test -e "$target"; then
    stat -c '%A %U:%G %s %y %n' "$target"
    du -sh "$target" 2>/dev/null
  else
    printf 'absent=%s\n' "$target"
  fi
done
if test -d "$FW/.git" || git -C "$FW" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$FW" rev-parse HEAD
  git -C "$FW" status --short --branch
  git -C "$FW" remote -v
fi
if test -x "$FENV/bin/python"; then
  "$FENV/bin/python" -V
  "$FENV/bin/python" -m pip list --format=freeze 2>/dev/null | grep -Ei '^(torch|torchvision|fastwam|numpy|scipy|packaging|pip|setuptools|wheel)=' | sort
fi
printf '%s\n' '-- FastWAM/install processes --'
ps -eo user,pid,ppid,stat,%cpu,%mem,rss,etimes,comm,args | grep -Ei 'fastwam|pip.*torch|conda.*fastwam' | grep -v -E 'grep -Ei|shenzhen_full_readonly_audit' || true
printf '%s\n' '-- selected pip cache entries --'
find "$FCACHE" -maxdepth 5 -type f \( -iname '*torch*' -o -iname '*.whl' -o -iname '*.part' \) -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -nr | head -n 60

section end
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds

exit 0
