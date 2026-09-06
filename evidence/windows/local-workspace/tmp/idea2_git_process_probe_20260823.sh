set -u
date '+%Y-%m-%d %H:%M:%S %Z'
ps -eo pid,ppid,pgid,stat,etime,args | grep -E 'git (push|ls-remote)|git-remote-https' | grep -v grep || true
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight status --short
timeout 8 git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight ls-remote personal refs/heads/codex/idea2-dvac-residual-downweight || true
