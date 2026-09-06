set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
run_id=idea2_dvac_sft_smoke_2gpu_16env_v1
runtime_dir=/root/autodl-tmp/idea2_dvac_runtime/$run_id
launch="$runtime_dir/launch.sh"
output="$target/outputs/$run_id"

test -f "$launch"
test ! -e "$output"
test ! -e "$runtime_dir/launcher.pid"
nohup bash "$launch" > "$runtime_dir/launcher.nohup.log" 2>&1 </dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$runtime_dir/launcher.pid"
sleep 2
test -d "/proc/$launcher_pid"
printf 'LAUNCHER_PID=%s\n' "$launcher_pid"
cat "$runtime_dir/status.env"
printf 'DETACHED_SMOKE_STARTED=1\n'

