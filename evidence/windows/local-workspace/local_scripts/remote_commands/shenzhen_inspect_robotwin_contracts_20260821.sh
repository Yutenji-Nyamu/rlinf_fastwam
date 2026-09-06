set -euo pipefail

robotwin_tree=/data/chenyiteng/projects/robotwin-native/RoboTwin
cd "$robotwin_tree"

for inspected_file in \
  scripts/update_xpolicylab.sh \
  scripts/requirements.txt \
  assets/_download.py \
  env_cfg/task_config/demo_clean.yml \
  env_cfg/eval/all_tasks.yml \
  XPolicyLab/policy/ACT/detr/process_data.py \
  XPolicyLab/utils/get_action_dim.sh \
  XPolicyLab/config/robot/aloha_agilex.yml \
  scripts/eval_policy.sh
do
  if [ ! -f "$inspected_file" ]; then
    printf '\n=== MISSING %s ===\n' "$inspected_file"
    continue
  fi
  printf '\n=== FILE %s ===\n' "$inspected_file"
  sha256sum "$inspected_file"
  wc -l "$inspected_file"
  sed -n '1,360p' "$inspected_file"
done

printf '\n=== RELEVANT CONFIG INVENTORY ===\n'
find XPolicyLab -maxdepth 4 -type f \
  \( -iname '*aloha*' -o -iname '*robot*info*' -o -iname '*action*dim*' \) \
  -printf '%P\n' | sort
