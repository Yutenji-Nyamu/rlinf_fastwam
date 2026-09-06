#!/usr/bin/env bash
set -u
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v1
printf '%s\n' 'driver failure context:'
sed -n '440,545p' "$RUN/runtime/driver.log"
printf '%s\n' 'resource extrema:'
awk -F, 'NR>1 && NF>=7 {if($4>m2)m2=$4;if($6>m3)m3=$6;if(min==0||$3<min)min=$3} END {printf "gpu2_peak_mib=%s gpu3_peak_mib=%s min_mem_available_gib=%.1f\n",m2,m3,min/1024/1024}' "$RUN/runtime/resource.csv"
printf '%s\n' 'owned actor/env process state:'
for p in 23399 23414 23434 23451 23420 23422; do ps -p "$p" -o pid=,stat=,cmd= 2>/dev/null || true; done
printf '%s\n' 'gpu2_3:'
nvidia-smi -i 2,3 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
