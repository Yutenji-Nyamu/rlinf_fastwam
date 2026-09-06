set -euo pipefail

ROOT=/data/chenyiteng/projects/fasterwam-standalone
REPO="$ROOT/FasterWAM-hustvl-official"

mkdir -p "$ROOT"
mkdir -p /data/chenyiteng/models/fasterwam
mkdir -p /data/chenyiteng/results/fasterwam-standalone

if [ -e "$REPO" ]; then
  printf 'TARGET_ALREADY_EXISTS %s\n' "$REPO"
  exit 2
fi

export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

git clone --recurse-submodules https://github.com/hustvl/FasterWAM.git "$REPO"

cd "$REPO"
printf 'SOURCE_HEAD=%s\n' "$(git rev-parse HEAD)"
printf 'SOURCE_BRANCH=%s\n' "$(git branch --show-current)"
git status --short
git submodule status --recursive
du -sh "$REPO"
