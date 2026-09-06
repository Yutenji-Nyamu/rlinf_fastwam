set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12
printf '%s\n' '--- terminal ---'
date --iso-8601=seconds
printf 'exit='; cat "$RUN/runtime/exit_code.txt" 2>/dev/null || true
ps -eo pid,ppid,etime,user,args | grep -E 'move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12|pi05_sidney_move_grpo_smoke1' | grep -v grep || true
nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo
printf '%s\n' '--- metrics.log ---'
cat "$RUN/metrics.log" 2>/dev/null || true
printf '%s\n' '--- driver key ---'
grep -Ein 'Global Step|train/|eval/|success|return|reward|filter|filtered|kl|clip|grad|loss|optimizer|micro|record|trajectory|Saving checkpoint|Saved checkpoint|checkpoint.*step|complete|Traceback|fatal|OOM|out of memory|ActorDied|exception' "$RUN/runtime/driver.log" | tail -n 260 || true
printf '%s\n' '--- driver final ---'
tail -n 180 "$RUN/runtime/driver.log" || true
printf '%s\n' '--- checkpoint tree ---'
find "$RUN" -path '*global_step_1*' -printf '%y %s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort || true
printf '%s\n' '--- checkpoint aggregate ---'
find "$RUN" -path '*global_step_1*' -type f -printf '%s\n' | awk '{n+=1;s+=$1} END{print "files="n,"bytes="s}'
printf '%s\n' '--- resource extrema ---'
awk '
  /^[0-9]{4}-/ {next}
  /^4,|^5,/ {gsub(/,/,"",$2); if($2+0>gmax[$1])gmax[$1]=$2+0; next}
  /^MemAvailable:/ {if(min==0||$2<min)min=$2}
  END {print "gpu4_peak_mib="gmax["4,"]; print "gpu5_peak_mib="gmax["5,"]; print "mem_available_min_kb="min}
' "$RUN/runtime/resources.log"
printf '%s\n' '--- fatal exact ---'
grep -Ein 'Traceback|fatal|CUDA out of memory|OutOfMemory|ActorDiedError|RayActorError|NCCL.*error|nonfinite|nan detected' "$RUN/runtime/driver.log" || true
