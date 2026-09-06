set -euo pipefail
for pid in 446490 446530 446500 446635 446515 446785; do
  printf '%s: ' "$pid"
  cat "/proc/$pid/wchan"
done
command -v py-spy || true
command -v strace || true
