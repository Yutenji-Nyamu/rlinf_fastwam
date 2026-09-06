#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
RLT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
DSRL=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin

printf '=== DRIVER_EXACT_LINES ===\n'
grep -a -n -E "Exception occurred while running EnvWorker|worker\(s\) were killed due to the node running low on memory|OOM kill reason|Ray killed 4 worker|Exiting main process due to a failure upon worker execution" "$RUN/driver.log" || true
first=$(grep -a -n -m1 "Exception occurred while running EnvWorker" "$RUN/driver.log" | cut -d: -f1)
if test -n "$first"; then
  start=$((first > 5 ? first - 5 : 1))
  end=$((first + 95))
  sed -n "${start},${end}p" "$RUN/driver.log"
fi

printf '=== RESOURCE_CSV_AROUND_EXIT ===\n'
/usr/bin/python3 - "$RUN/resource.csv" <<'PY'
import csv
import datetime as dt
import json
import sys

path = sys.argv[1]
with open(path, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None

valid = [r for r in rows if fnum(r, 'host_mem_available_kib') is not None]
cvalid = [r for r in valid if fnum(r, 'cgroup_memory_current_bytes') is not None]
minimum = min(valid, key=lambda r: fnum(r, 'host_mem_available_kib'))
maximum_cg = max(cvalid, key=lambda r: fnum(r, 'cgroup_memory_current_bytes'))
print(json.dumps({
    'rows': len(rows),
    'first_timestamp': rows[0]['timestamp'],
    'last_timestamp': rows[-1]['timestamp'],
    'minimum_host_available_timestamp': minimum['timestamp'],
    'minimum_host_available_gib': round(fnum(minimum, 'host_mem_available_kib') / 2**20, 3),
    'maximum_driver_cgroup_timestamp': maximum_cg['timestamp'],
    'maximum_driver_cgroup_gib': round(fnum(maximum_cg, 'cgroup_memory_current_bytes') / 2**30, 3),
}, sort_keys=True))
for row in rows:
    stamp = row['timestamp']
    if '2026-08-23T11:54:' <= stamp <= '2026-08-23T12:03:':
        payload = {
            'timestamp': stamp,
            'driver_alive': row.get('driver_alive'),
            'host_available_gib': None if fnum(row, 'host_mem_available_kib') is None else round(fnum(row, 'host_mem_available_kib') / 2**20, 3),
            'driver_cgroup_gib': None if fnum(row, 'cgroup_memory_current_bytes') is None else round(fnum(row, 'cgroup_memory_current_bytes') / 2**30, 3),
        }
        print(json.dumps(payload, sort_keys=True))
PY

printf '=== CHECK_TIMESTAMPS ===\n'
for item in "RLT:$RLT" "DSRL:$DSRL"; do
  name=${item%%:*}; path=${item#*:}
  printf '%s head=' "$name"; git -C "$path" rev-parse HEAD
  printf '%s commit=' "$name"; git -C "$path" show -s --format='%cI %s' HEAD
  printf '%s pytest-cache-files\n' "$name"
  find "$path/.pytest_cache" -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 30 || true
  printf '%s pycache-window-count=' "$name"
  find "$path" -xdev -type f -path '*/__pycache__/*' -newermt '2026-08-23 11:50:00 UTC' ! -newermt '2026-08-23 12:05:00 UTC' 2>/dev/null | wc -l
  find "$path" -xdev -type f -path '*/__pycache__/*' -newermt '2026-08-23 11:50:00 UTC' ! -newermt '2026-08-23 12:05:00 UTC' \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 20 || true
done

printf '=== RAY_SESSIONS_EXACT ===\n'
find /tmp/ray -mindepth 1 -maxdepth 1 -type d -name 'session_*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort

printf 'SZ_GRPO_OOM_TIMELINE_NARROW_OK\n'
