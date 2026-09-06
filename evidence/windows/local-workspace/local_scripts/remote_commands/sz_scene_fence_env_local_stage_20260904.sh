#!/usr/bin/env bash
set -euo pipefail
R=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
cd "$R"
test "$(git rev-parse HEAD)" = b60144fd60270f303fb5ea229ca92125f6b0a710
git apply --check /home/chenyiteng/tmp/scene_fence_env_init_only_20260904.patch
for file in build.sh smoke.py README.md; do
  diff -u --label "a/tools/fastwam_scene_fence/$file" --label "b/tools/fastwam_scene_fence/$file" "tools/fastwam_scene_fence/$file" "/home/chenyiteng/tmp/scene-fence-local-$file" > "/home/chenyiteng/tmp/scene-fence-local-$file.diff" || test "$?" = 1
  git apply --check "/home/chenyiteng/tmp/scene-fence-local-$file.diff"
done
git apply /home/chenyiteng/tmp/scene_fence_env_init_only_20260904.patch
for file in build.sh smoke.py README.md; do git apply "/home/chenyiteng/tmp/scene-fence-local-$file.diff"; done
git diff --check
printf 'export RLINF_SCENE_FENCE_LIBRARY=%q\n' /home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/librlinf_scene_fence.so > /home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/enable.sh
source /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1/runtime/environment.sh
"$VIRTUAL_ENV/bin/python" tools/fastwam_scene_fence/smoke.py --prepare --source-config /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1/runtime/resolved.yaml --output /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/scene-fence-env-local-smoke-20260904
date -Is
