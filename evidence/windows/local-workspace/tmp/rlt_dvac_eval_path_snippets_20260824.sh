set -u
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
printf 'RUNNER_EVAL\n'
sed -n '185,212p' "$repo/rlinf/runners/embodied_runner.py"
printf 'MODEL_GET_ACTION\n'
sed -n '675,750p' "$repo/rlinf/models/embodiment/openpi/openpi_action_model.py"
printf 'ROLLOUT_EVALUATE_DEFS\n'
grep -RIn "def evaluate" "$repo/rlinf/workers/rollout" | head -20
sed -n '745,805p' "$repo/rlinf/workers/rollout/hf/huggingface_worker.py"
