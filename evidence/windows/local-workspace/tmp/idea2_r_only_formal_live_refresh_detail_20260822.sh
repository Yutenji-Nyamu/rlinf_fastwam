#!/usr/bin/env bash
set -u

run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822

printf 'TIME_AND_PHASE\n'
date --iso-8601=seconds
tail -n 80 "$runtime_dir/driver.log" 2>/dev/null || true

printf 'TRACEBACK_CONTEXT\n'
grep -an -B 8 -A 28 'Traceback (most recent call last)' "$runtime_dir/driver.log" 2>/dev/null || true

printf 'CHECKPOINT_TREE\n'
find "$run_dir" -type d -name 'global_step_*' -printf '%p\n' 2>/dev/null | sort -V
while IFS= read -r d; do
  du -sh "$d"
  printf '%s\tfiles=' "$d"
  find "$d" -type f | wc -l
done < <(find "$run_dir" -type d -name 'global_step_*' -print 2>/dev/null | sort -V)

printf 'CGROUP_LIMITS\n'
for f in memory.current memory.max memory.high memory.events memory.events.local memory.stat; do
  printf 'FILE=/sys/fs/cgroup/%s\n' "$f"
  if [[ -f "/sys/fs/cgroup/$f" ]]; then cat "/sys/fs/cgroup/$f"; else printf 'missing\n'; fi
done

printf 'RESOURCE_EXTREMA\n'
python - "$runtime_dir/resource_monitor/resources.csv" <<'PY'
import csv, sys
p=sys.argv[1]
rows=list(csv.DictReader(open(p, newline='')))
print('rows',len(rows),'first',rows[0]['timestamp'],'last',rows[-1]['timestamp'])
for k in ['gpu_memory_used_mib','gpu_util_pct','cgroup_current_bytes','cgroup_anon_bytes','cgroup_file_bytes','host_mem_available_kib','disk_available_kib']:
    vals=[float(r[k]) for r in rows if r.get(k) not in ('',None)]
    print(k,'min',min(vals),'max',max(vals),'last',vals[-1])
for k in ['event_low','event_high','event_max','event_oom','event_oom_kill']:
    vals=[int(float(r[k])) for r in rows if r.get(k) not in ('',None)]
    print(k,'first',vals[0],'max',max(vals),'last',vals[-1])
for gpu in sorted(set(r['gpu_index'] for r in rows)):
    subset=[r for r in rows if r['gpu_index']==gpu]
    print('gpu',gpu,'mem_peak_mib',max(int(float(r['gpu_memory_used_mib'])) for r in subset),'util_mean',sum(float(r['gpu_util_pct']) for r in subset)/len(subset),'samples',len(subset))
PY

printf 'METRICS_FILE_LAST_BLOCK\n'
tail -n 80 "$run_dir/metrics.log" 2>/dev/null || true

printf 'STATUS_AFTER\n'
date --iso-8601=seconds
for name in wrapper driver observer; do
  pid=$(cat "$runtime_dir/${name}.pid" 2>/dev/null || true)
  [[ -n "$pid" && -d "/proc/$pid" ]] && printf '%s=%s:alive\n' "$name" "$pid" || printf '%s=%s:dead_or_missing\n' "$name" "${pid:-missing}"
done
