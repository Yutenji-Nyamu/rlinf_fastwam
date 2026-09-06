set -u
source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
date '+%Y-%m-%d %H:%M:%S %Z'
git -C "$source_root" rev-parse HEAD
git -C "$source_root" status --short
git -C "$source_root" ls-remote personal refs/heads/codex/idea2-dvac-residual-downweight || true
pgrep -af 'git push personal HEAD:codex/idea2-dvac-residual-downweight' || true
