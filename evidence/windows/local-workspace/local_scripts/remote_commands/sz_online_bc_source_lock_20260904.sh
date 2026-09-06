#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1
base=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
target=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
pin=dc9b87cc49334c7516487ead68ebeb060fd7c090
date -Is
if ! git -C "$base" cat-file -e "$pin^{commit}" 2>/dev/null; then
  timeout 45s git -C "$base" -c http.proxy=http://127.0.0.1:7890 -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=20 fetch origin "$pin"
fi
test ! -e "$target"
git -C "$base" worktree add -b codex/sz-pi0-online-bc "$target" "$pin"
git -C "$target" rev-parse HEAD
git -C "$target" status --short
sed -n '1,240p' "$target/examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml"
find /data/chenyiteng/models -maxdepth 3 -type d -iname '*pi0*' -print
find /data/chenyiteng/projects/rlinf-shenzhen -maxdepth 2 -name '*environment*.sh' -print
sed -n '35,260p' /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support/robotwin/envs/vector_env.py
