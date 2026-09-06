set -euo pipefail
native_parent=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
native="$native_parent/third_party/RoboTwin"
rlrt=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
wt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
run=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1

echo SOURCE_IDENTITIES
git -C "$native_parent" rev-parse HEAD
git -C "$native_parent" ls-tree HEAD third_party/RoboTwin || true
git -C "$rlrt" rev-parse HEAD
git -C "$wt" rev-parse HEAD

echo NATIVE_INSTRUCTION
grep -nE 'task_description|instruction_type|get_instruction|instruction' "$native/envs/_base_task.py" "$native/robotwin/envs/vector_env.py" "$native/robotwin/envs/task_env.py" 2>/dev/null | head -120 || true

echo RLRT_INSTRUCTION
grep -nE 'task_description|instruction_type|get_instruction|instruction' "$rlrt/robotwin/envs/vector_env.py" "$rlrt/robotwin/envs/task_env.py" 2>/dev/null | head -160 || true

echo CURRENT_RESOLVED
resolved=$(find "$run" -type f -name 'config.yaml' -o -name 'resolved.yaml' 2>/dev/null | head -1)
if test -n "$resolved"; then
  echo "$resolved"
  grep -nE 'task_name:|max_episode_steps:|max_steps_per_rollout_epoch:|step_lim:|group_size:|noise_level:|center_crop:|random_background:|cluttered_table:|clean_background_rate:|random_head_camera_dis:|random_table_height:|random_light:|crazy_random_light_rate:|seed:|use_fixed_reset_state_ids:|num_steps:|action_chunk:|config_name:' "$resolved" | head -220
fi

echo SEED_BANK_HEAD
python - "$wt/rlinf/envs/robotwin/seeds/train_seeds.json" "$wt/rlinf/envs/robotwin/seeds/eval_seeds.json" <<'PY'
import json,sys
for path in sys.argv[1:]:
    d=json.load(open(path))
    s=d['move_stapler_pad'].get('success_seeds')
    print(path, len(s) if s else None, s[:40] if s else None)
PY
