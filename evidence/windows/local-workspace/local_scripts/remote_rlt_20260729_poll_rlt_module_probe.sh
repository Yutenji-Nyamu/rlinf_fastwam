set -eu

EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_pre_smoke_20260729
PID_PATH="$EVIDENCE_ROOT/robotwin_rlt_prefix_probe_with_rlt.pid"
STDOUT_PATH="$EVIDENCE_ROOT/robotwin_rlt_prefix_probe_with_rlt.json"
STDERR_PATH="$EVIDENCE_ROOT/robotwin_rlt_prefix_probe_with_rlt.stderr"

probe_pid=$(cat "$PID_PATH")
if kill -0 "$probe_pid" 2>/dev/null; then
  status=RUNNING
else
  status=EXITED
fi

printf 'STATUS=%s\nPID=%s\n' "$status" "$probe_pid"
printf '%s\n' '--- GPU ---'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader || true
printf '%s\n' '--- STDOUT ---'
tail -n 160 "$STDOUT_PATH" 2>/dev/null || true
printf '%s\n' '--- STDERR ---'
tail -n 160 "$STDERR_PATH" 2>/dev/null || true
