set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
cd "$root"
test "$(git rev-parse HEAD)" = 700b6846dbc2fe02398de05c044c8097cc974774
test "$(git branch --show-current)" = codex/sz-pi0-online-bc
git diff --check
git diff --stat
git add -- rlinf/envs/robotwin/robotwin_env.py examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml tests/unit_tests/test_online_bc.py
git commit -m "Set online BC U10 and preserve fixed evaluation seeds in two batches"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 90s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/codex/sz-pi0-online-bc
git rev-parse HEAD
git status --porcelain
