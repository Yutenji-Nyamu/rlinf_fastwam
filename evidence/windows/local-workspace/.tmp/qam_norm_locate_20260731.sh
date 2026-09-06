set -eu

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
model_dir=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle

grep -nE 'model_path|model_name|checkpoint|norm|stats' \
  "$repo/examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml"
find "$model_dir" -maxdepth 6 -type f \
  \( -name 'norm_stats.json' -o -name 'model.safetensors.index.json' \) \
  -print
