set -eu
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney
P="$ROOT/packets/parity-m10-phys4-v3"
A="$ROOT/packets/b1-adjust-badseed-retry-m10-phys4-v7"
B="$ROOT/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v7"
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
printf '%s\n' '=== SOURCE ==='
git -C "$WT" rev-parse HEAD
if test -z "$(git -C "$WT" status --porcelain)"; then echo clean; else git -C "$WT" status --short; fi
printf '%s\n' '=== LIVE GPU ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' '=== GPU4/5 APPS ==='
nvidia-smi -i 4,5 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
printf '%s\n' '=== RAM/DISK/RAY ==='
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo
df -h /data /home
RAY_ADDRESS=172.17.0.1:6389 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ray status 2>&1 | sed -n '1,30p'
printf '%s\n' '=== PACKETS ==='
for d in "$P" "$A" "$B"; do
  printf 'packet=%s complete=' "$d"
  tr '\n' ' ' < "$d/packet-complete.txt"
  printf ' head='
  tr '\n' ' ' < "$d/source-head.txt"
  echo
  cat "$d/contract.json"
done
printf '%s\n' '=== PARITY COMMANDS ==='
cat "$P/commands.txt"
printf '%s\n' '=== B1 COMMAND ==='
cat "$A/command.txt"
printf '%s\n' '=== ONE-STEP COMMAND ==='
cat "$B/command.txt"
