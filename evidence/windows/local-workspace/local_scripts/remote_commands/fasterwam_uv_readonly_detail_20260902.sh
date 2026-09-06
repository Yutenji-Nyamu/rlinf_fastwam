set -u

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
CACHE=/data/chenyiteng/cache/uv-fasterwam
pid="$(pgrep -f '/home/chenyiteng/miniforge3/bin/uv sync .*FasterWAM-hustvl-official/environments/robotwin' | head -n 1)"
date '+TIME %Y-%m-%d %H:%M:%S %Z'
if test -z "$pid"; then
  printf 'UV_NOT_RUNNING\n'
  exit 0
fi

printf 'UV_PID %s\n' "$pid"
ps -p "$pid" -o pid,ppid,etimes,state,%cpu,%mem,time,wchan:32,args
printf 'WCHAN '
cat "/proc/$pid/wchan" 2>/dev/null || true
printf '\nIO\n'
cat "/proc/$pid/io" 2>/dev/null || true

printf 'OPEN_FDS_RELEVANT\n'
for fd in /proc/"$pid"/fd/*; do
  target="$(readlink "$fd" 2>/dev/null || true)"
  case "$target" in
    *uv-fasterwam*|*FasterWAM*|*\.whl*|*\.tmp*|socket:*)
      printf '%s -> %s\n' "$(basename "$fd")" "$target"
      ;;
  esac
done | tail -n 80

printf 'NETWORK_SOCKETS\n'
ss -tpn 2>/dev/null | grep -E "pid=$pid,|users:\(\(\"uv\",pid=$pid," || true

printf 'CACHE_NEWEST_FILES\n'
find "$CACHE" -type f -printf '%T@ %s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -n | tail -n 25

printf 'CACHE_LARGEST_FILES\n'
find "$CACHE" -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -n | tail -n 15

printf 'VENV_TREE\n'
du -sh "$REPO/.venvs/robotwin" 2>/dev/null || true
find "$REPO/.venvs/robotwin" -maxdepth 3 -type f -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -n 20
