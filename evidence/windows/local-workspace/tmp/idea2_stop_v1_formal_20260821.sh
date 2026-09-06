#!/usr/bin/env bash
set -euo pipefail

DRIVER=114149
RUN=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
EXPERIMENT="$RUN/idea2_dvac_apply_formal_100step_2gpu16env_20260821"
RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821
EXPECTED=/root/autodl-tmp/RLinf_idea2_dvac_train/examples/embodiment/train_embodied_agent.py

cmdline=$(tr '\0' ' ' < "/proc/$DRIVER/cmdline")
case "$cmdline" in
  *"$EXPECTED"*) ;;
  *) echo "DRIVER_IDENTITY_MISMATCH: $cmdline"; exit 2 ;;
esac

latest=$(find "$EXPERIMENT/checkpoints" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sort -V | tail -n 1)
checkpoint="$EXPERIMENT/checkpoints/$latest"
test -d "$checkpoint"
echo "LATEST_CHECKPOINT=$latest"
echo "CHECKPOINT_FILES=$(find "$checkpoint" -type f | wc -l)"
echo "CHECKPOINT_BYTES=$(du -sb "$checkpoint" | awk '{print $1}')"
echo "CHECKPOINT_PARTIALS=$(find "$checkpoint" -type f -name '*.partial' | wc -l)"

date --iso-8601=seconds > "$RUNTIME/stop_requested_at.txt"
echo "SIGINT_DRIVER=$DRIVER"
kill -INT "$DRIVER"
