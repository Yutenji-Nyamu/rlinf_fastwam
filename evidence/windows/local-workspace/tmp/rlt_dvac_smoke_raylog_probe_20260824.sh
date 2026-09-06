set -u
printf 'RAY_LOG_FILES\n'
find /tmp/ray/session_latest/logs -maxdepth 1 -type f \( -name '*7876*' -o -name '*7894*' -o -name '*7898*' -o -name '*7900*' \) -printf '%s %p\n' | sort -n
printf 'RAY_LOG_TAILS\n'
for f in $(find /tmp/ray/session_latest/logs -maxdepth 1 -type f \( -name '*7876*' -o -name '*7894*' \) | sort); do
  printf 'FILE=%s\n' "$f"
  tail -80 "$f"
done
