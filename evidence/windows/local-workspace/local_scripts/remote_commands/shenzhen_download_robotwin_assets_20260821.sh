set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

robotwin_tree=/data/chenyiteng/projects/robotwin-native/RoboTwin
cd "$robotwin_tree"

export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30

printf '%s\n' '=== PRECHECK ==='
test "$(git rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
test "$(git -C XPolicyLab rev-parse HEAD)" = c07a09614dd44cc4a67483bcb9a82e7439d99926
for asset_target in \
  assets/background_texture.zip assets/embodiments.zip assets/objects.zip \
  assets/background_texture assets/embodiments assets/objects
do
  if [ -e "$asset_target" ]; then
    printf 'REFUSE_EXISTING_ASSET_TARGET=%s\n' "$asset_target" >&2
    exit 40
  fi
done
printf 'https_proxy=%s\n' "${https_proxy:-unset}"
df -h "$robotwin_tree"
du -sh assets

printf '%s\n' '=== OFFICIAL ASSET DOWNLOAD AND EXTRACTION ==='
timeout --signal=TERM --kill-after=60s 7200 bash scripts/_download_assets.sh

printf '%s\n' '=== VERIFY ASSETS ==='
for removed_archive in assets/background_texture.zip assets/embodiments.zip assets/objects.zip; do
  test ! -e "$removed_archive"
  printf 'official_script_removed=%s\n' "$removed_archive"
done
for extracted_dir in assets/background_texture assets/embodiments assets/objects; do
  test -d "$extracted_dir"
  du -sh "$extracted_dir"
  find "$extracted_dir" -type f | wc -l
done
du -sh assets
find assets -maxdepth 2 -type d -printf '%P\n' | sort | sed -n '1,120p'
df -h "$robotwin_tree"
git status --short --branch
