#!/usr/bin/env bash
set -euo pipefail
OLD=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1
OUT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/scene-fence-env-local-smoke-20260904
test -s "$OLD/runtime/stopped_for_scene_fence_20260904.json"
test ! -e "$OUT/smoke.log"
test -z "$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
kill -0 321933 322685 3176215
source "$OLD/runtime/environment.sh"
source /home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/enable.sh
export CUDA_VISIBLE_DEVICES=6
cd "$REPO_PATH"
date -Is
set +e
timeout --signal=TERM --kill-after=15s 300s env LD_DEBUG=bindings LD_DEBUG_OUTPUT="$OUT/bindings" "$VIRTUAL_ENV/bin/python" tools/fastwam_scene_fence/smoke.py --source-config "$OLD/runtime/resolved.yaml" --output "$OUT" > "$OUT/smoke.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$OUT/exit_code"
tail -n 45 "$OUT/smoke.log"
grep 'binding file .*libsvulkan2.so.*librlinf_scene_fence.so.*_ZN8svulkan28renderer10RTRenderer6render' "$OUT"/bindings.* > "$OUT/native_binding.txt" || true
cat "$OUT/native_binding.txt"
test -s "$OUT/native_binding.txt"
kill -0 321933 322685 3176215
nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
date -Is
exit "$code"
