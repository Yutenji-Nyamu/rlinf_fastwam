set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
expected_head=6d0db56bf26f972cd27fa29535f5eb939e80e5bf
expected_branch=codex/idea2-dvac-pi0-robotwin

test "$(git -C "$target" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$target" branch --show-current)" = "$expected_branch"
test -z "$(git -C "$target" status --porcelain)"
git -C "$target" apply --check --whitespace=error-all -
printf 'PATCH_CHECK=PASS\n'
