set -euo pipefail

uid="$(id -u)"
matches=()
while read -r pid; do
  [ -n "$pid" ] || continue
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  case "$cmd" in
    *"collect_data.sh adjust_bottle sz_collect_smoke_1ep_20260821 0"*)
      matches+=("$pid")
      printf 'MATCH pid=%s cmd=%s\n' "$pid" "$cmd"
      ;;
  esac
done < <(pgrep -u "$uid" -x timeout || true)

test "${#matches[@]}" -eq 1
pid="${matches[0]}"
kill -TERM "$pid"
sleep 3
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL "$pid"
fi

printf '%s\n' '=== remaining related processes ==='
ps -u "$uid" -o pid,ppid,pgid,stat,cmd | \
  grep -E 'collect_data|sz_collect_smoke_1ep_20260821|adjust_bottle.py' || true
