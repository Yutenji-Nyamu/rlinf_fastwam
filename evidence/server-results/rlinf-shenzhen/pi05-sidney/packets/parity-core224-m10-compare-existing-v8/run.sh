#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/parity-core224-m10-compare-existing-v8
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-core224-m10-compare-existing-v8
test "$(git -C "$WT" rev-parse HEAD)" = 639444db8ad7bc9c934e056ad8511139ed94eba9
test -z "$(git -C "$WT" status --porcelain)"
set +e
bash -e "$PACKET/command.txt" > "$RUN/runtime/wrapper.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN/runtime/exit_code.txt"
exit "$RC"
