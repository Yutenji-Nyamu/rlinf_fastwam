set -eu
cd /root/autodl-tmp/RLinf_fastwam_rlinf
(git diff --name-only; git ls-files --others --exclude-standard) | sort -u | xargs sha256sum
cat .gitignore
cat examples/embodiment/config/robotwin_adjust_bottle_grpo_fastwam_a800_2gpu.yaml
cat examples/embodiment/config/model/fastwam_robotwin.yaml

if test -d /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40/.git; then
  git -C /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40 rev-parse HEAD
  git -C /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40 branch --show-current
  git -C /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40 status --short --branch
  git -C /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40 remote -v
else
  printf '%s\n' 'WAMPPO_NOT_GIT'
fi
