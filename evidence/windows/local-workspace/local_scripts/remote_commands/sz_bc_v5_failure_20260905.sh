#!/usr/bin/env bash
set -eu
id
date -Is
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
run=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v5
git -C "$root" rev-parse HEAD
git -C "$root" status --short
tail -n 230 "$run/driver.log"
find "$run" -maxdepth 3 -type f -name '*log' -printf '%p %s\n'
