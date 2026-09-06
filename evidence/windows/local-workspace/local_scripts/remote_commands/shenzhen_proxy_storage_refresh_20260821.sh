#!/usr/bin/env bash

set +e

date --iso-8601=seconds
sha256sum /home/readme_to_codex.md /home/readme_storage_to_codex.md /home/readme_network_to_codex.md 2>&1
ls -l /home/chenyiteng/data /home/chenyiteng/shared /home/chenyiteng/scratch 2>&1
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
findmnt -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS / /home /data 2>&1
grep -vE '^[[:space:]]*(#|$)' /etc/fstab 2>&1
stat -c '%A %U:%G %s %n' /etc/profile.d/mihomo-proxy.sh 2>&1
sed -n '1,160p' /etc/profile.d/mihomo-proxy.sh 2>&1
source /etc/profile.d/mihomo-proxy.sh 2>/dev/null
env | grep -iE '(^|_)(http|https|all|no)_proxy=' | sort
ss -ltn 2>&1 | grep -E '(^State|127\.0\.0\.1:7890)'
printf '[proxy hf]\n'
timeout 25s curl -fsSIL --max-time 20 https://huggingface.co/robots.txt 2>&1 | sed -n '1,16p'
printf '[direct hf]\n'
timeout 15s curl --noproxy '*' -fsSIL --max-time 10 https://huggingface.co/robots.txt 2>&1 | sed -n '1,16p'
printf '[proxy github git]\n'
timeout 25s git ls-remote https://github.com/RoboTwin-Platform/RoboTwin.git HEAD refs/heads/main 2>&1
printf '[proxy hf model metadata]\n'
timeout 25s curl -fsSIL --max-time 20 https://huggingface.co/api/models/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle 2>&1 | sed -n '1,16p'
date --iso-8601=seconds
exit 0
