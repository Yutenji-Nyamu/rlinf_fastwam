set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
test "$(git rev-parse HEAD)" = dc3711950a2cad8222ba72bfe0c6de2f7a5babdb
test "$(git status --short)" = ' M rlinf/workers/actor/fsdp_qam_policy_worker.py'
git diff --check
git diff -- rlinf/workers/actor/fsdp_qam_policy_worker.py
git add -- rlinf/workers/actor/fsdp_qam_policy_worker.py
git commit -m "fix(qam): keep prefix drift diagnostic nonblocking"
git status --short
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
