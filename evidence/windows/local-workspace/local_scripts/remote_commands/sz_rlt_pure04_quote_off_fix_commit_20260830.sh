#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421
branch=codex/sz-rlt-dvac-pure-single-gpu
expected_head=220b415bbe384e47a44bc91d3c61502f84c13c87
config=examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_control.yaml
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

test "$(git -C "$worktree" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$worktree" branch --show-current)" = "$branch"
test -z "$(git -C "$worktree" status --porcelain)"

git -C "$worktree" apply - <<'PATCH'
diff --git a/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_control.yaml b/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_control.yaml
--- a/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_control.yaml
+++ b/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_control.yaml
@@ -24,6 +24,6 @@ actor:
 algorithm:
   rlt_dvac:
-    mode: off
+    mode: "off"
   rlt_schedule:
     warmup_min_size: 20000
   replay_buffer:
PATCH

git -C "$worktree" diff --check
"$venv/bin/python" -c 'from omegaconf import OmegaConf; import sys; c=OmegaConf.load(sys.argv[1]); assert c.algorithm.rlt_dvac.mode == "off" and isinstance(c.algorithm.rlt_dvac.mode, str)' "$worktree/$config"
git -C "$worktree" add -- "$config"
git -C "$worktree" commit -m 'fix(rlt): quote disabled DVAC mode in YAML'
git -C "$worktree" push personal "HEAD:refs/heads/$branch"

head=$(git -C "$worktree" rev-parse HEAD)
test "$head" = "$(git -C "$worktree" rev-parse "personal/$branch")"
test -z "$(git -C "$worktree" status --porcelain)"
printf 'head=%s\nremote_head=%s\nQUOTE_OFF_FIX_OK\n' "$head" "$(git -C "$worktree" rev-parse "personal/$branch")"
