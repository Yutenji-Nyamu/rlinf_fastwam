set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv
NAME=fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2
RUN="$ROOT/runs/$NAME"
RELOAD="$ROOT/runs/${NAME}-reload-step2"
PACKET="$ROOT/packets/$NAME"
pid=$(cat "$PACKET/worker.pid")
printf 'time=%s worker_pid=%s alive=%s mem_available_gib=%s\n' \
  "$(date --iso-8601=seconds)" "$pid" "$([ -d "/proc/$pid" ] && echo yes || echo no)" \
  "$(awk '/^MemAvailable:/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)"
nvidia-smi -i 2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
for dir in "$RUN" "$RELOAD"; do
  printf 'run=%s exit=%s\n' "$dir" "$(cat "$dir/runtime/exit_code.txt" 2>/dev/null || echo pending)"
  log="$dir/runtime/driver.log"
  if [ -f "$log" ]; then
    grep -E 'Global Step|eval.*success|rollout.*success|actor/dvac|Resuming training|Traceback|Error|OutOfMemory|NCCL|Gloo' "$log" | tail -n 18 || true
    tail -n 12 "$log" || true
    printf 'log_bytes=%s\n' "$(stat -c %s "$log")"
  fi
done
printf 'marker_ok=%s marker_failed=%s worker_exit=%s\n' \
  "$([ -f "$PACKET/SMOKE_OK" ] && echo yes || echo no)" \
  "$([ -f "$PACKET/SMOKE_FAILED" ] && echo yes || echo no)" \
  "$(cat "$PACKET/worker_exit_code.txt" 2>/dev/null || echo pending)"
find "$RUN" "$RELOAD" -type f \( -name '.metadata' -o -name '*.distcp' -o -name 'dvac_state_rank*.json' -o -name 'runner_step_*.pt' -o -name 'weight_summary.json' \) 2>/dev/null | sort || true
