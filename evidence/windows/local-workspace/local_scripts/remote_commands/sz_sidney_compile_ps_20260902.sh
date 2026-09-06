set -euo pipefail
ps -eo pid,ppid,etime,stat,wchan:24,cmd | grep -E 'run_sz_verified|compileall|git.*diff|less|lerobot-v060' | grep -v grep || true
