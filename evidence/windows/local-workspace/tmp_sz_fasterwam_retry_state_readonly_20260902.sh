set -u
date --iso-8601=seconds
echo '=== RETRY WRAPPER TREE ==='
for p in 79696 79697; do
  if [ -d "/proc/$p" ]; then
    ps -p "$p" -o user=,pid=,ppid=,etime=,%cpu=,%mem=,rss=,stat=,cmd=
    printf 'pid%s stdout=' "$p"; readlink "/proc/$p/fd/1" || true
    printf 'pid%s stderr=' "$p"; readlink "/proc/$p/fd/2" || true
  else
    printf 'PID_GONE %s\n' "$p"
  fi
done
children=$(pgrep -P 79697 2>/dev/null | tr '\n' ' ')
printf 'children=%s\n' "$children"
for p in $children; do
  ps -p "$p" -o user=,pid=,ppid=,etime=,%cpu=,%mem=,rss=,stat=,cmd= 2>/dev/null || true
  printf 'child%s stdout=' "$p"; readlink "/proc/$p/fd/1" 2>/dev/null || true
done
echo '=== POSSIBLE LOG TAIL ==='
for p in 79696 79697; do
  [ -d "/proc/$p" ] || continue
  out=$(readlink "/proc/$p/fd/1" 2>/dev/null || true)
  case "$out" in
    /*) [ -f "$out" ] && tail -n 80 "$out" 2>/dev/null || true ;;
  esac
done
echo '=== FILES ==='
find /data/chenyiteng/models/fasterwam/release-6bf9471 -maxdepth 5 -type f \
  \( -name '*.incomplete' -o -name 'step_029355.pt' -o -name 'dataset_stats.json' \) \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -k3
