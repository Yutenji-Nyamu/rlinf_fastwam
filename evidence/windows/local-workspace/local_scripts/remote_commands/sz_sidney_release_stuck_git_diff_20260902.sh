set -euo pipefail
pid=1392846
cmd=$(ps -p "$pid" -o args= || true)
case "$cmd" in
  "/usr/bin/pager") kill -TERM "$pid" ;;
  *) echo "refuse unexpected pid/cmd: $pid $cmd" >&2; exit 2 ;;
esac
