set -u
for pid in 7876 7894; do
  printf 'PID=%s\n' "$pid"
  for f in /tmp/ray/session_latest/logs/*"$pid"*.out /tmp/ray/session_latest/logs/*"$pid"*.err; do
    [ -f "$f" ] || continue
    printf 'FILE=%s SIZE=%s MTIME=%s\n' "$f" "$(stat -c %s "$f")" "$(stat -c %y "$f")"
    tail -120 "$f"
  done
done
