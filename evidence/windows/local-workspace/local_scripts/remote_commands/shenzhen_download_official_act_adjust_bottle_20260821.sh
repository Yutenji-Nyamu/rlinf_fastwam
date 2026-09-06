set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

target=/data/chenyiteng/models/robotwin2-hf-a967b852
leaf="$target/act_ckpt/act-adjust_bottle/demo_clean-50"
revision=a967b852afa21a9cbf19a198f7e653109042e87c

if [ -e "$target" ]; then
  printf 'REFUSE_EXISTING_MODEL_TARGET=%s\n' "$target" >&2
  exit 40
fi

printf '%s\n' '=== PRECHECK ==='
printf 'https_proxy=%s\n' "${https_proxy:-unset}"
df -h /data/chenyiteng

export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export HF_REVISION="$revision"

printf '%s\n' '=== SELECTIVE OFFICIAL ACT CHECKPOINT DOWNLOAD ==='
timeout --signal=TERM --kill-after=60s 1800s python - <<'PY'
import os
from huggingface_hub import snapshot_download

result = snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    repo_type="dataset",
    revision=os.environ["HF_REVISION"],
    allow_patterns=[
        "act_ckpt/act-adjust_bottle/demo_clean-50/policy_last.ckpt",
        "act_ckpt/act-adjust_bottle/demo_clean-50/dataset_stats.pkl",
    ],
    local_dir="/data/chenyiteng/models/robotwin2-hf-a967b852",
    max_workers=1,
)
print("snapshot_dir=", result)
PY

printf '%s\n' '=== VERIFY OFFICIAL ACT CHECKPOINT ==='
test "$(stat --printf='%s' "$leaf/policy_last.ckpt")" = 335907442
test "$(stat --printf='%s' "$leaf/dataset_stats.pkl")" = 10664
printf '%s  %s\n' \
  edfb0125103e67465cc2852ea1683acc2ce1060d02b81ba6f1113b5420b40690 \
  "$leaf/policy_last.ckpt" | sha256sum --check -
printf '%s  %s\n' \
  a79964a7cce7a02cd172fb669ef12c8c2f5cedd2c4bd11adfd86c4d90cb239c4 \
  "$leaf/dataset_stats.pkl" | sha256sum --check -
find "$target" -type f -printf '%s\t%p\n' | sort -nr
du -sh "$target"
df -h /data/chenyiteng
