set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

root=/data/chenyiteng/projects/robotwin-native/RoboTwin
target="$root/data/demo_clean/adjust_bottle"
archive="$root/data/download_cache/dataset/adjust_bottle/demo_clean.zip"

cd "$root"
test "$(git rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
test "$(git -C XPolicyLab rev-parse HEAD)" = c07a09614dd44cc4a67483bcb9a82e7439d99926

printf '%s\n' '=== PRECHECK ==='
if [ -e "$target" ] || [ -e "$archive" ]; then
  printf 'REFUSE_EXISTING_DATA_TARGET target=%s archive=%s\n' "$target" "$archive" >&2
  exit 40
fi
printf 'https_proxy=%s\n' "${https_proxy:-unset}"
df -h "$root"

export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export HF_KEEP_ARCHIVES=1
export HF_REVISION=a967b852afa21a9cbf19a198f7e653109042e87c
export HF_MAX_WORKERS=1
export HF_EXTRACT_WORKERS=1

printf '%s\n' '=== OFFICIAL SINGLE-TASK DOWNLOAD AND EXTRACTION ==='
timeout --signal=TERM --kill-after=60s 1800 \
  bash scripts/download_xpolicylab_data.sh adjust_bottle

printf '%s\n' '=== VERIFY SINGLE-TASK DATA ==='
test -f "$archive"
test -d "$target/aloha_agilex/data"
stat --printf='archive_bytes=%s\n' "$archive"
printf 'hdf5_count=%s\n' "$(find "$target/aloha_agilex/data" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l)"
printf 'video_count=%s\n' "$(find "$target/aloha_agilex/video" -maxdepth 1 -type f | wc -l)"
printf 'instruction_count=%s\n' "$(find "$target/aloha_agilex/instruction" -maxdepth 1 -type f | wc -l)"
du -sh "$archive" "$target"
df -h "$root"
git status --short --branch
