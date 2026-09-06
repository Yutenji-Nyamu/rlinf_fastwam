set -u

CACHE=/data/chenyiteng/cache/uv-fasterwam
date '+TIME %Y-%m-%d %H:%M:%S %Z'
pid="$(pgrep -f '/home/chenyiteng/miniforge3/bin/uv sync .*FasterWAM-hustvl-official/environments/robotwin' | head -n 1)"
printf 'UV_PID %s\n' "${pid:-none}"
test -n "$pid" && { ps -p "$pid" -o pid,etimes,%cpu,time,state,wchan:28; cat "/proc/$pid/io" 2>/dev/null; } || true

printf 'TOP_TEMP_FILES\n'
find "$CACHE" -type f -path "$CACHE/.tmp*" -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -nr | head -n 12

printf 'TEMP_DIR_TOTALS\n'
for path in "$CACHE"/.tmp*; do
  test -d "$path" || continue
  du -sb "$path" 2>/dev/null
done | sort -nr | head -n 12
