set -euo pipefail

REVISION=6bf9471ced6919a15ab8fded89f7772f5060c44b
TARGET=/data/chenyiteng/models/fasterwam/release-6bf9471
PYTHON=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python

mkdir -p "$TARGET" /data/chenyiteng/cache/huggingface
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export HF_HOME=/data/chenyiteng/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0

download_ok=0
for attempt in 1 2 3 4 5 6; do
    printf 'DOWNLOAD_ATTEMPT %s\n' "$attempt"
    if "$PYTHON" - <<PY
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="hustvl/FasterWAM",
    revision="$REVISION",
    allow_patterns=["robotwin/*"],
    local_dir="$TARGET",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("SNAPSHOT_PATH", path)
PY
    then
        download_ok=1
        break
    fi
    sleep 5
done
test "$download_ok" = 1

unset HTTP_PROXY HTTPS_PROXY

CKPT="$TARGET/robotwin/step_029355.pt"
STATS="$TARGET/robotwin/dataset_stats.json"
test -s "$CKPT"
test -s "$STATS"
printf 'RELEASE_FILES\n'
stat -c '%s %n' "$CKPT" "$STATS"
sha256sum "$CKPT" "$STATS"
printf 'RELEASE_TOTAL\n'
du -sh "$TARGET"
