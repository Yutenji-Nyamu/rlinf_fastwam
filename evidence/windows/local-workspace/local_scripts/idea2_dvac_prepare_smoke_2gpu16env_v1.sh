set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
runtime_dir=/root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1
config=/root/autodl-tmp/idea2_dvac_run_configs/idea2_dvac_sft_smoke_2gpu_16env_v1.yaml
output="$target/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1"
head=61996e15cc7f5a32bd6012b61b20893d94636c82

test "$(git -C "$target" rev-parse HEAD)" = "$head"
test "$(git -C "$target" rev-parse personal/codex/idea2-dvac-pi0-robotwin)" = "$head"
test -z "$(git -C "$target" status --porcelain)"
test ! -e "$config"
test ! -e "$output"
test ! -e "$runtime_dir"
mkdir -p "$runtime_dir"
printf 'RUNTIME_DIR=%s\nCONFIG_STATE=ABSENT\nOUTPUT_STATE=ABSENT\nPREPARE_SMOKE16_PASS=1\n' \
  "$runtime_dir"

