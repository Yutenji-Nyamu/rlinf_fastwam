set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
cd "$RLT_ROOT"

git add -A
git diff --cached --check
git commit -m "feat(rlt): port RoboTwin pi0 RL-token training"
git rev-parse HEAD
git status --short
