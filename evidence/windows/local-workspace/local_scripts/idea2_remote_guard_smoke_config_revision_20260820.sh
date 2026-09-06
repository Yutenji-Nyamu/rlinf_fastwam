set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
branch=codex/idea2-dvac-pi0-robotwin
head=61996e15cc7f5a32bd6012b61b20893d94636c82
config=/root/autodl-tmp/idea2_dvac_run_configs/idea2_dvac_sft_smoke_2gpu_2env_v1.yaml
old_hash=4f8e66ab634d08c6922bbe30d07b380587133b45c8b63f7eaff17d7c3d468b16
output=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/outputs/idea2_dvac_sft_smoke_2gpu_2env_v1

test "$(git -C "$target" rev-parse HEAD)" = "$head"
test "$(git -C "$target" rev-parse personal/$branch)" = "$head"
test -z "$(git -C "$target" status --porcelain)"
test -f "$config"
test "$(sha256sum "$config" | cut -d' ' -f1)" = "$old_hash"
test ! -e "$output"
printf 'SOURCE_AND_REMOTE=%s\nOLD_CONFIG_HASH=%s\nOUTPUT_STATE=ABSENT\nCONFIG_REVISION_ALLOWED=1\n' \
  "$head" "$old_hash"

