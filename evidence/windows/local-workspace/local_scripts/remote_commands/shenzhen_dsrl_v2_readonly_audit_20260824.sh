#!/usr/bin/env bash
set -u

root=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2
run="$root/run"
log="$run/driver.log"

printf '%s\n' '--- time ---'
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds

printf '%s\n' '--- owner ---'
pid=$(cat "$run/wrapper.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state=alive; else state=dead; fi
printf 'wrapper_pid=%s state=%s\n' "${pid:-missing}" "$state"
if [ -n "$pid" ]; then ps -o pid,ppid,pgid,lstart,etime,rss,vsz,stat,cmd -p "$pid" --no-headers 2>/dev/null || true; fi

printf '%s\n' '--- log inventory ---'
for f in "$log" "$run/resource.csv" "$run/resolved.yaml" "$run/command.txt" "$run/exit_code.txt"; do
  if [ -e "$f" ]; then stat -c '%n|bytes=%s|mtime=%y' "$f"; else printf '%s|missing\n' "$f"; fi
done
printf 'resource_rows='; wc -l < "$run/resource.csv" 2>/dev/null || true
printf 'resource_header='; head -n 1 "$run/resource.csv" 2>/dev/null || true
printf 'driver_global_step_tables='; tr '\r' '\n' < "$log" 2>/dev/null | grep -c 'Global Step:' || true

printf '%s\n' '--- resolved algorithm/runtime subset ---'
grep -nE 'max_steps:|rollout_epoch:|num_envs:|eval_interval:|save_interval:|warmup|update_epoch:|global_batch_size:|micro_batch_size:|capacity:|execution_chunk_size:|dsrl_eval_deterministic:|experiment_name:|log_path:|ckpt_path:|save_path:|video_base_dir:' "$run/resolved.yaml" 2>/dev/null | head -n 120 || true

printf '%s\n' '--- latest progress and metrics ---'
tr '\r' '\n' < "$log" 2>/dev/null \
  | grep -E 'Global Step:|success_once=|success_at_end=|reward=|resident_transitions=|total_inserted=|global_resident_transitions=|planned_optimizer_updates=|update_step=|critic|actor|entropy|alpha|Saving checkpoint|Evaluat|eval/' \
  | tail -n 180 || true

printf '%s\n' '--- fatal/error scan counts ---'
for pat in 'Traceback' 'CUDA out of memory' 'OutOfMemory' 'WorkerCrashed' 'RayTaskError' 'nonfinite' 'nan' 'NCCL' 'ERROR'; do
  n=$(tr '\r' '\n' < "$log" 2>/dev/null | grep -i -c "$pat" || true)
  printf '%s=%s\n' "$pat" "$n"
done

printf '%s\n' '--- checkpoint and event files ---'
find "$root" -type f \( -path '*/checkpoints/*' -o -name 'events.out.tfevents*' -o -name '*metrics*.json*' -o -name '*.pt' -o -name '*.safetensors' \) \
  -printf '%T@|%s|%p\n' 2>/dev/null | sort -n | tail -n 160 || true
printf '%s\n' '--- checkpoint directories ---'
find "$root" -type d -path '*/checkpoints/global_step_*' -printf '%T@|%p\n' 2>/dev/null | sort -n || true

printf '%s\n' '--- output tree size/count ---'
du -sh "$root" "$root/robotwin_data/train" "$root/robotwin_data/eval" "$root/video/train" "$root/video/eval" 2>/dev/null || true
for d in "$root/robotwin_data/train" "$root/robotwin_data/eval" "$root/video/train" "$root/video/eval"; do
  if [ -d "$d" ]; then printf '%s|files=' "$d"; find "$d" -type f 2>/dev/null | wc -l; fi
done

printf '%s\n' '--- gpu and memory ---'
nvidia-smi -i 6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits
free -b | sed -n '1,2p'
df -B1 /data | tail -n 1
tail -n 5 "$run/resource.csv" 2>/dev/null || true
