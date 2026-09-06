#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
export PATH=/home/chenyiteng/miniforge3/bin:/home/chenyiteng/.local/bin:$PATH

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RUN=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821
TMPROOT=/home/chenyiteng/tmp/rlinf-7d07-install-20260821
TOKEN=/home/chenyiteng/.cache/openpi/big_vision/paligemma_tokenizer.model
TOKEN_SHA=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6

test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test -z "$(git -C "$ROOT" status --porcelain)"
test ! -e "$VENV"
test ! -e "$RUN"
test ! -e "$TMPROOT"
test -x /home/chenyiteng/miniforge3/bin/pip
test -s /etc/vulkan/icd.d/nvidia_icd.json
test "$(stat -c %s "$TOKEN")" = 4264023
printf '%s  %s\n' "$TOKEN_SHA" "$TOKEN" | sha256sum --check -
test -f /home/chenyiteng/.cache/openpi/paligemma_tokenizer.model

mkdir -p "$RUN" "$TMPROOT"
export TMPDIR="$TMPROOT"

printf '%s\n' '=== OFFICIAL INSTALL COMMAND ==='
printf '%q ' bash requirements/install.sh embodied --model openpi --env robotwin \
  --venv "$VENV" --no-root
printf '\n'
printf 'start_time=%s\n' "$(date --iso-8601=seconds)"

cd "$ROOT"
timeout --signal=INT --kill-after=120s 10800s \
  bash requirements/install.sh embodied \
    --model openpi \
    --env robotwin \
    --venv "$VENV" \
    --no-root \
  2>&1 | tee "$RUN/install.log"

printf 'end_time=%s\n' "$(date --iso-8601=seconds)"
test -x "$VENV/bin/python"
test -s "$VENV/bin/activate"
printf '%s\n' '=== BASIC POST-INSTALL MANIFEST ==='
"$VENV/bin/python" --version
"$VENV/bin/python" -c 'import torch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'
command -v uv
uv --version
du -sh "$VENV" /home/chenyiteng/.cache/uv /home/chenyiteng/.cache/openpi "$TMPROOT" 2>/dev/null || true
df -h / /home /data
printf 'worktree_status='; git -C "$ROOT" status --short --branch
printf '%s\n' 'R1_OFFICIAL_INSTALL_OK'
