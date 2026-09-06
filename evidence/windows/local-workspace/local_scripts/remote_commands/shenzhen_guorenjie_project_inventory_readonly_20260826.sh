#!/usr/bin/env bash
set -euo pipefail

sudo -S -p '' true

echo '===== admin_boundary ====='
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
id guorenjie
sudo -n -l -U guorenjie 2>&1 || true
stat -c '%A %U:%G %s %n' /home/guorenjie

sudo -n -u guorenjie bash <<'INNER'
set -euo pipefail
export HOME=/home/guorenjie
export GIT_OPTIONAL_LOCKS=0
PROJ=/home/guorenjie/research/smolvla-libero-clp
REPO="$PROJ/lerobot"

echo '===== process_topology ====='
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nu)
for pid in $pids; do
  [[ -r "/proc/$pid/status" ]] || continue
  owner=$(stat -c %U "/proc/$pid")
  [[ "$owner" == guorenjie ]] || continue
  ps -o pid=,ppid=,pgid=,sid=,lstart=,etimes=,%cpu=,%mem=,rss=,stat=,comm=,args= -p "$pid"
  printf 'cwd='; readlink "/proc/$pid/cwd" || true
  printf 'stdout='; readlink "/proc/$pid/fd/1" || true
  printf 'stderr='; readlink "/proc/$pid/fd/2" || true
  printf 'cgroup='; tr '\n' ' ' < "/proc/$pid/cgroup"; echo
  printf 'selected_env='; tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^(CUDA_VISIBLE_DEVICES|LOCAL_RANK|RANK|WORLD_SIZE|MASTER_ADDR|MASTER_PORT|HF_HOME|HF_HUB_CACHE|HF_LEROBOT_HOME|LEROBOT_HOME|XDG_CACHE_HOME)=' | tr '\n' ';' || true; echo
done
echo '-- launchers --'
ps -eo pid=,ppid=,pgid=,sid=,lstart=,etimes=,%cpu=,%mem=,rss=,stat=,comm=,args= --sort=pid \
  | awk '/guorenjie\/.conda\/envs\/lerobot-clp\/bin\/(torchrun|lerobot-train)/ {print}' || true

echo '===== project_git_and_size ====='
stat -c '%A %U:%G %s %y %n' "$PROJ"
du -sh "$PROJ" "$PROJ/outputs" 2>/dev/null || true
echo '-- top-level --'; find "$PROJ" -maxdepth 2 -mindepth 1 -not -path "$PROJ/.git/*" -printf '%y %s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort | head -n 180
git -C "$REPO" rev-parse --show-toplevel
printf 'head='; git -C "$REPO" rev-parse HEAD
printf 'branch='; git -C "$REPO" branch --show-current
echo '-- remotes (names only) --'; git -C "$REPO" remote || true
echo '-- recent commits --'; git -C "$REPO" log -5 --date=iso --pretty='format:%h %ad %an %s'
echo '-- status --'; git -C "$REPO" status --short --untracked-files=normal || true

echo '===== relevant_source_names ====='
find "$PROJ" -path "$REPO/.git" -prune -o -path "$PROJ/outputs" -prune -o -type f \
  \( -iname 'README*' -o -iname '*clp*' -o -iname '*train*' -o -iname '*eval*' -o -iname '*.yaml' -o -iname '*.yml' \) \
  -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort -k3 | head -n 220

echo '===== outputs_inventory ====='
find "$PROJ/outputs/train" -mindepth 1 -maxdepth 1 -type d -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort
for out in "$PROJ"/outputs/train/pi0_libero_fullft_rel_vis_*_s30k; do
  [[ -d "$out" ]] || continue
  echo "-- $out --"
  du -sh "$out"
  find "$out" -maxdepth 3 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort -k2 | tail -n 100
done

echo '===== dataset_and_cache_inventory ====='
for root in /home/guorenjie/.cache /home/guorenjie/.cache/huggingface /home/guorenjie/.cache/huggingface/lerobot /home/guorenjie/.cache/huggingface/hub; do
  [[ -e "$root" ]] && du -sh "$root" 2>/dev/null || true
done
find /home/guorenjie/.cache /home/guorenjie/research -maxdepth 7 \
  \( -iname '*libero*' -o -iname 'info.json' -o -iname 'stats.json' -o -iname 'tasks.jsonl' -o -iname 'episodes.jsonl' \) \
  -printf '%y %s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort | head -n 240

echo '===== user_network_and_storage ====='
ss -tpn 2>/dev/null | grep -E 'users:\(\("(python3\.12|lerobot-train|torchrun)"' || true
df -hT /home /data
echo SZ_GUORENJIE_PROJECT_INVENTORY_OK
INNER
