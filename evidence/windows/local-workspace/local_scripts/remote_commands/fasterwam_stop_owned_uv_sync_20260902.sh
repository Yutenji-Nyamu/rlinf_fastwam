set -euo pipefail

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
mapfile -t pids < <(ps -eo pid=,comm=,args= | awk -v repo="$REPO" '$2 == "uv" && index($0, repo) && index($0, "environments/robotwin") {print $1}')
test "${#pids[@]}" -eq 1
pid="${pids[0]}"
cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
case "$cmdline" in
  *uv*sync*"$REPO"*environments/robotwin*) ;;
  *) printf 'Refusing unexpected PID %s: %s\n' "$pid" "$cmdline" >&2; exit 24 ;;
esac
printf 'TERMINATING_OWNED_UV %s %s\n' "$pid" "$cmdline"
kill -TERM "$pid"
for _ in 1 2 3 4 5; do
  if ! kill -0 "$pid" 2>/dev/null; then
    printf 'OWNED_UV_STOPPED %s\n' "$pid"
    exit 0
  fi
  sleep 1
done
printf 'Owned uv did not exit after TERM: %s\n' "$pid" >&2
exit 25
