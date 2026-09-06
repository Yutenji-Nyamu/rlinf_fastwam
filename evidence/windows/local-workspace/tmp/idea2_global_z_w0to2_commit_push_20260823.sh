set -euo pipefail
source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
relative=examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_global_z_w0to2_100step_formal.yaml
cd "$source_root"
echo '=== pre_commit ==='
git status --short
test "$(git status --short | wc -l)" -eq 1
git status --short | grep -q "?? $relative"
git add -- "$relative"
git commit -m 'config: add global-z DVAC zero-to-two formal run'
commit=$(git rev-parse HEAD)
echo "COMMIT=$commit"
git push personal HEAD:codex/idea2-dvac-residual-downweight
echo '=== post_push ==='
git status --short
git ls-remote personal refs/heads/codex/idea2-dvac-residual-downweight
