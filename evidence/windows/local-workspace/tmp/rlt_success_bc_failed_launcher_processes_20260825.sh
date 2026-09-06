set -euo pipefail
ps -eo pid,ppid,pgid,stat,cmd | awk '$2==418626 || $1==418626 {print}'
