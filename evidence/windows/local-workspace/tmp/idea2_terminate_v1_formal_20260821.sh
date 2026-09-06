#!/usr/bin/env bash
set -euo pipefail

DRIVER=114149
RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821
EXPECTED=/root/autodl-tmp/RLinf_idea2_dvac_train/examples/embodiment/train_embodied_agent.py

cmdline=$(tr '\0' ' ' < "/proc/$DRIVER/cmdline")
case "$cmdline" in
  *"$EXPECTED"*) ;;
  *) echo "DRIVER_IDENTITY_MISMATCH: $cmdline"; exit 2 ;;
esac

date --iso-8601=seconds > "$RUNTIME/terminate_requested_at.txt"
echo "SIGTERM_DRIVER=$DRIVER"
kill -TERM "$DRIVER"
