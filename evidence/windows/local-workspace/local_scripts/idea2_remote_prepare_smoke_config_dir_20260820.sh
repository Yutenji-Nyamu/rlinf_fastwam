set -euo pipefail

config_dir=/root/autodl-tmp/idea2_dvac_run_configs
config_path="$config_dir/idea2_dvac_sft_smoke_2gpu_2env_v1.yaml"
test ! -e "$config_path"
install -d -m 0755 "$config_dir"
test ! -e "$config_path"
printf 'SMOKE_CONFIG_DIR_READY=%s\n' "$config_dir"
