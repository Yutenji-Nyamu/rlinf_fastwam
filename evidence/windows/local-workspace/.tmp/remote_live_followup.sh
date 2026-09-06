set -u
date '+%F %T %Z'
REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN="$REPO/logs/20260718_100910-robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu"
printf 'driver='; pgrep -f "$REPO/examples/embodiment/[t]rain_embodied_agent.py" | head -n 1 || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
tail -n 140 "$RUN/run_embodiment.log" 2>/dev/null || true
echo '=== PEAK ==='
cat "$RUN/resource_monitor/peak.txt" 2>/dev/null || true
echo '=== CHECKPOINTS ==='
find "$RUN" -maxdepth 5 -type d -name 'global_step_*' -print 2>/dev/null | sort || true
