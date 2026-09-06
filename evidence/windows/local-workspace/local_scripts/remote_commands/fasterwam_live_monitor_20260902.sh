set -u

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
TARGET=/data/chenyiteng/models/fasterwam/release-6bf9471

date '+TIME %Y-%m-%d %H:%M:%S %Z'
printf 'PROCESSES\n'
ps -eo pid,ppid,etimes,stat,%cpu,%mem,args | grep -E 'install_robotwin|uv sync|curobo|hf-mirror.com/hustvl/FasterWAM|step_029355|wget' | grep -v grep || true

printf 'DOWNLOAD_FILES\n'
if test -d "$TARGET"; then
  find "$TARGET" -type f \( -name 'step_029355.pt' -o -name '*.incomplete' -o -name 'dataset_stats.json' \) -printf '%s %p\n' 2>/dev/null | sort -n
  du -sh "$TARGET" 2>/dev/null || true
else
  printf 'TARGET_ABSENT\n'
fi

printf 'ROBOTWIN_ENV\n'
if test -x "$REPO/.venvs/robotwin/bin/python"; then
  stat -c 'PYTHON %s %y %n' "$REPO/.venvs/robotwin/bin/python"
  du -sh "$REPO/.venvs/robotwin" 2>/dev/null || true
else
  printf 'PYTHON_ABSENT\n'
  test -d "$REPO/.venvs/robotwin" && du -sh "$REPO/.venvs/robotwin" 2>/dev/null || true
fi

printf 'UV_CACHE\n'
test -d /data/chenyiteng/cache/uv-fasterwam && du -sh /data/chenyiteng/cache/uv-fasterwam 2>/dev/null || printf 'UV_CACHE_ABSENT\n'

printf 'GPU3\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,pstate --format=csv,noheader,nounits | sed -n '4p'
