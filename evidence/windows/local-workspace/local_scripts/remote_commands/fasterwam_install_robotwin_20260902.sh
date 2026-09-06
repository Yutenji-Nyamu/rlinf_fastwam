set -euo pipefail

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
ASSET_SOURCE=/data/chenyiteng/projects/robotwin-native/RoboTwin/assets
ASSET_DEST="$REPO/third_party/RoboTwin/assets"

test "$(git -C "$REPO" rev-parse HEAD)" = 83667817df0d4f823f39d90700e61ea2f432ac45
for name in background_texture embodiments objects; do
    source_path="$ASSET_SOURCE/$name"
    dest_path="$ASSET_DEST/$name"
    test -d "$source_path"
    mkdir -p "$ASSET_DEST"
    if test -L "$dest_path"; then
        test "$(readlink -f "$dest_path")" = "$(readlink -f "$source_path")"
    elif test -e "$dest_path"; then
        printf 'Unexpected existing asset destination: %s\n' "$dest_path" >&2
        exit 22
    else
        ln -s "$source_path" "$dest_path"
    fi
done

export PATH=/home/chenyiteng/miniforge3/bin:$PATH
export UV_CACHE_DIR=/data/chenyiteng/cache/uv-fasterwam
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=3

cd "$REPO"
bash scripts/setup/install_robotwin.sh

printf 'ROBOTWIN_ENV_RESULT\n'
.venvs/robotwin/bin/python - <<'PY'
import numpy
import torch
import hydra
import sapien

print('python_ok')
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('numpy', numpy.__version__)
print('sapien', sapien.__version__ if hasattr(sapien, '__version__') else 'imported')
PY
printf 'ASSET_LINKS\n'
for name in background_texture embodiments objects; do
    printf '%s -> %s\n' "$ASSET_DEST/$name" "$(readlink -f "$ASSET_DEST/$name")"
done
printf 'ENV_SIZE\n'
du -sh .venvs/robotwin
