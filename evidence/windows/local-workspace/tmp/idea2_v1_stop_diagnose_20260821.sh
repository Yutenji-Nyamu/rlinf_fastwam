#!/usr/bin/env bash
set -u

DRIVER=114149
RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821

date --iso-8601=seconds
ps -p "$DRIVER" -o pid=,stat=,etimes=,cmd=
grep -E '^(State|SigPnd|ShdPnd|SigBlk|SigIgn|SigCgt):' "/proc/$DRIVER/status"
tail -n 80 "$RUNTIME/driver.log"
test -f "$RUNTIME/driver.exitcode" && cat "$RUNTIME/driver.exitcode"
