set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh

robotwin_parent=/data/chenyiteng/projects/robotwin-native
robotwin_tree="$robotwin_parent/RoboTwin"
robotwin_commit=30954692d06ba7e89f7a6b76064f4062c488fa81

printf '%s\n' '=== PRECHECK ==='
id
printf 'cwd='; pwd
printf 'proxy='; printf '%s\n' "${https_proxy:-unset}"
df -h /data/chenyiteng
stat -c 'target-parent=%n owner=%U:%G mode=%a' /data/chenyiteng
if [ -e "$robotwin_tree" ]; then
  printf 'REFUSE_EXISTING_TARGET=%s\n' "$robotwin_tree" >&2
  exit 40
fi

printf '%s\n' '=== REMOTE LOCK ==='
git ls-remote https://github.com/RoboTwin-Platform/RoboTwin.git refs/heads/main
curl -fsSL --max-time 30 https://api.github.com/repos/RoboTwin-Platform/RoboTwin \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("github_repo_size_kib=", d.get("size")); print("default_branch=", d.get("default_branch"))'

printf '%s\n' '=== CLONE ==='
mkdir -p "$robotwin_parent"
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git "$robotwin_tree"
git -C "$robotwin_tree" checkout --detach "$robotwin_commit"
git -C "$robotwin_tree" submodule sync --recursive
git -C "$robotwin_tree" submodule update --init --recursive

printf '%s\n' '=== VERIFY ==='
printf 'robotwin_head='; git -C "$robotwin_tree" rev-parse HEAD
printf 'robotwin_branch='; git -C "$robotwin_tree" branch --show-current
git -C "$robotwin_tree" status --short --branch
git -C "$robotwin_tree" submodule status --recursive
du -sh "$robotwin_tree"
find "$robotwin_tree" -maxdepth 2 -type d -printf '%P\n' | sort | sed -n '1,120p'
