set -u

runtime_dir=/root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1
output=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1
printf 'STATUS_AT=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' STATUS_ENV_BEGIN
cat "$runtime_dir/status.env" 2>/dev/null || true
printf '%s\n' STATUS_ENV_END
printf '%s\n' PROCESS_BEGIN
ps -p 21052,21072,21073 -o pid=,ppid=,stat=,etime=,rss=,comm=,args= 2>/dev/null || true
printf '%s\n' PROCESS_END
printf '%s\n' GPU_BEGIN
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
printf '%s\n' GPU_END
if test -f "$runtime_dir/resource_monitor/resources.csv"; then
  printf 'RESOURCE_GPU_ROWS=%s\n' "$(( $(wc -l < "$runtime_dir/resource_monitor/resources.csv") - 1 ))"
  tail -n 2 "$runtime_dir/resource_monitor/resources.csv"
fi
printf 'OUTPUT_STATE=%s\n' "$(if test -d "$output"; then printf PRESENT; else printf ABSENT; fi)"
printf '%s\n' DRIVER_TAIL_BEGIN
tail -n 35 "$runtime_dir/driver.log" 2>/dev/null || true
printf '%s\n' DRIVER_TAIL_END
