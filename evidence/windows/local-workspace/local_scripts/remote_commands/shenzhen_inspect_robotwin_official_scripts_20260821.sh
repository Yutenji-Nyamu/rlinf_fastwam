set -euo pipefail

robotwin_tree=/data/chenyiteng/projects/robotwin-native/RoboTwin
cd "$robotwin_tree"

printf '%s\n' '=== SOURCE LOCK ==='
git rev-parse HEAD
git status --short --branch
git submodule status --recursive

printf '%s\n' '=== TOP-LEVEL FILE INVENTORY ==='
find scripts -maxdepth 1 -type f -printf '%f\n' | sort
find XPolicyLab/policy/ACT -maxdepth 3 -type f -printf '%P\n' | sort

for inspected_file in \
  scripts/_install.sh \
  scripts/_download_assets.sh \
  scripts/download_xpolicylab_data.sh \
  XPolicyLab/policy/ACT/README.md \
  XPolicyLab/policy/ACT/install.sh \
  XPolicyLab/policy/ACT/process_data.sh \
  XPolicyLab/policy/ACT/train.sh \
  XPolicyLab/policy/ACT/eval.sh
do
  printf '\n=== FILE %s ===\n' "$inspected_file"
  sha256sum "$inspected_file"
  wc -l "$inspected_file"
  sed -n '1,260p' "$inspected_file"
done
