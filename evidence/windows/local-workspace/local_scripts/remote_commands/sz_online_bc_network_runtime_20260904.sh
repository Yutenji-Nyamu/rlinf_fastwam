#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
date -Is
ss -ltn | grep -E '789|909|1080|8888|6389'
for url in https://github.com/RLinf/RLinf.git/info/refs?service=git-upload-pack https://api.github.com/repos/RLinf/RLinf/commits/dc9b87cc49334c7516487ead68ebeb060fd7c090; do
  curl -LIsS --max-time 10 "$url" | head -n 4
done
root=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
git -C "$root" rev-parse HEAD
git -C "$root" status --short
grep -nE 'n_steps_to_run|def step|success|execut' "$root/envs/vector_env.py" | head -n 65
sed -n '35,245p' "$root/envs/vector_env.py"
find /data/chenyiteng/models -maxdepth 3 -type d -iname '*pi0*' -print
find /data/chenyiteng/projects/rlinf-shenzhen -maxdepth 2 -name '*environment*.sh' -print
find /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages -maxdepth 1 -name '*.pth' -exec grep -H 'robotwin\|openpi\|RLinf' {} \;
