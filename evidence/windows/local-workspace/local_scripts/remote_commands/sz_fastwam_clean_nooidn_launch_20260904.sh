#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
RUN=$ROOT/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1
DONOR=$ROOT/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-clean-oidn-off-20260904
test "$(id -un)" = chenyiteng
test "$(git -C "$RT" rev-parse HEAD)" = f3e30a83365cdda1165911422dc3ce73e703201e
test -z "$(git -C "$RT" status --porcelain)"
test ! -e "$RUN/runtime/wrapper.pid"
test ! -e "$RUN/runtime/driver.log"
test -s "$RUN/runtime/wrapper.sh"
test -s "$DONOR/runtime/observer.sh"
bash -n "$RUN/runtime/wrapper.sh"
bash -n "$RUN/runtime/environment.sh"
source "$RUN/runtime/environment.sh"
test "$(git -C "$REPO_PATH" rev-parse HEAD)" = 4faade1d50bf21d1caf1b8a4e5f89282a810208a
test -z "$(git -C "$REPO_PATH" status --porcelain)"
cd "$RUN/runtime"
sha256sum -c sha256.txt
test -z "$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
ps -p 321933,322685,3176215 -o user,pid,etime,stat,comm
kill -0 321933 322685 3176215
"$VIRTUAL_ENV/bin/python" - <<'PY'
import os,ray,collections
ray.init(address=os.environ['RAY_ADDRESS'],namespace='codex_fastwam_clean_nooidn_prelaunch',logging_level='ERROR')
counts=collections.Counter(r['namespace'] for r in ray.util.list_named_actors(all_namespaces=True))
print('PRELAUNCH_NAMESPACES',dict(counts))
assert counts['RLinf']>0 and counts['RLinf_1']==0,counts
ray.shutdown()
PY
cp "$DONOR/runtime/observer.sh" "$RUN/runtime/observer.sh"
chmod 700 "$RUN/runtime/wrapper.sh" "$RUN/runtime/observer.sh"
nohup setsid bash "$RUN/runtime/wrapper.sh" "$RUN/runtime" > "$RUN/runtime/wrapper.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN/runtime/wrapper.pid"
printf '%s\n' "$pid" > "$RUN/runtime/owned.pgid"
nohup setsid bash "$RUN/runtime/observer.sh" "$pid" "$RUN/runtime/resource.csv" > "$RUN/runtime/observer.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN/runtime/observer.pid"
sleep 5
kill -0 "$pid"
date -Is
printf 'NEW_RUN=%s\nWRAPPER_PID=%s\n' "$RUN" "$pid"
ps -p "$pid" -o user,pid,pgid,stat,args
tail -n 12 "$RUN/runtime/driver.log"
