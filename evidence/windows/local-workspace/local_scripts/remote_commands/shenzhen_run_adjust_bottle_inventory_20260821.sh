set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

root=/data/chenyiteng/projects/robotwin-native/RoboTwin

cd "$root"
test "$(git rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
test "$(git -C XPolicyLab rev-parse HEAD)" = c07a09614dd44cc4a67483bcb9a82e7439d99926
python -
du -sh data/demo_clean/adjust_bottle data/download_cache/dataset/adjust_bottle/demo_clean.zip
df -h "$root"
