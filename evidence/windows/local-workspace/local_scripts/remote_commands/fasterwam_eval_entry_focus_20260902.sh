set -euo pipefail

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
cd "$REPO"

printf 'SINGLE_ENTRY_KEYS\n'
grep -nE "eval_num_episodes|task_config|output_dir|eval_policy|seed|instruction_type|dataset_stats|subprocess|CUDA_VISIBLE" experiments/robotwin/eval_robotwin_single.py
printf 'SINGLE_ENTRY_BODY\n'
sed -n '100,300p' experiments/robotwin/eval_robotwin_single.py
printf 'ROBOTWIN_EVAL_POLICY_KEYS\n'
grep -nE "eval_num|episode_num|seed|task_config|instruction_type|save_video|policy|args\." third_party/RoboTwin/script/eval_policy.py
printf 'ROBOTWIN_EVAL_POLICY_BODY\n'
sed -n '1,340p' third_party/RoboTwin/script/eval_policy.py
printf 'DEPLOY_POLICY_YAML\n'
sed -n '1,260p' experiments/robotwin/fasterwam_policy/deploy_policy.yml
printf 'TASK_STEP_LIMIT\n'
grep -n "move_stapler_pad" third_party/RoboTwin/task_config/_eval_step_limit.yml
printf 'ASSET_DIRS\n'
for p in third_party/RoboTwin/assets/background_texture third_party/RoboTwin/assets/embodiments third_party/RoboTwin/assets/objects; do
  if [ -d "$p" ]; then du -sh "$p"; else printf 'MISSING %s\n' "$p"; fi
done
