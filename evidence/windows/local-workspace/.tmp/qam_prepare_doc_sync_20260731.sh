set -eu

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
test "$(git branch --show-current)" = codex/qam-pi0-robotwin
test -z "$(git status --short)"
mkdir -p docs/rlinf-robotwin-pi0-qam/evidence local_scripts
echo "READY $(git rev-parse HEAD)"

