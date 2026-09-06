#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
OLD=$ROOT/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1
RUN=$ROOT/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3
test "$(id -un)" = chenyiteng
test ! -e "$RUN/runtime/wrapper.pid"
test ! -e "$RUN/runtime/driver.log"
source "$OLD/runtime/environment.sh"
test "$(git -C "$REPO_PATH" rev-parse HEAD)" = "$(cat "$RUN/runtime/source_head.txt")"
test -z "$(git -C "$REPO_PATH" status --porcelain)"
test "$(git -C "$ROBOTWIN_PATH" rev-parse HEAD)" = f3e30a83365cdda1165911422dc3ce73e703201e
test -z "$(git -C "$ROBOTWIN_PATH" status --porcelain)"
cd "$RUN/runtime"
sha256sum -c sha256.txt
test "$(sha256sum /home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/librlinf_scene_fence.so | cut -d' ' -f1)" = 45e6cac35ea2edea0724fa4cd92ec82cab401b6abd6f8001805d22306f375df0
test -z "$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
kill -0 321933 322685 3176215
"$VIRTUAL_ENV/bin/python" - <<'PY'
import ray, collections, os
ray.init(address=os.environ['RAY_ADDRESS'],namespace='codex_scene_fence_prelaunch',logging_level='ERROR')
c=collections.Counter(r['namespace'] for r in ray.util.list_named_actors(all_namespaces=True))
print('PRELAUNCH_NAMESPACES',dict(c))
assert c['RLinf']>0 and c['RLinf_1']==0,c
ray.shutdown()
PY
bash -n "$RUN/runtime/environment.sh"
bash -n "$RUN/runtime/wrapper.sh"
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
tail -n 15 "$RUN/runtime/driver.log"
