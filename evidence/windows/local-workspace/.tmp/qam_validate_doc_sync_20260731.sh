set -eu

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"

/root/autodl-tmp/RLinf/.venv/bin/python -m py_compile \
  local_scripts/remote_exec_autodl.py
bash -n \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_qonly_smoke_launch_20260731_v1.sh
git diff --check

if grep -R -n -F 'NBo7SQqoatnZ' \
  PROJECT_CONTEXT.md HANDOFF.md local_scripts/remote_exec_autodl.py \
  docs/rlinf-robotwin-pi0-qam; then
  echo "CREDENTIAL_SCAN_FAILED" >&2
  exit 1
fi

git status --short
git diff --stat
echo "HEAD=$(git rev-parse HEAD)"

