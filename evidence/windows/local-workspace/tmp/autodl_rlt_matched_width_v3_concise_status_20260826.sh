#!/usr/bin/env bash
set -euo pipefail

control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_matched_width_formal480_20260826_v3
method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3
control_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_matched_width_formal480_20260826_v3/runtime
method_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3/runtime
pair_rt=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_matched_width_formal480_20260826_v3
python=/root/autodl-tmp/RLinf/.venv/bin/python

echo TIME
date -Is
echo LIFECYCLE
for label in control method; do
  test "$label" = control && rt=$control_rt || rt=$method_rt
  pid=$(cat "$rt/wrapper.pid")
  printf '%s pid=%s alive=' "$label" "$pid"
  kill -0 "$pid" 2>/dev/null && echo yes || echo no
  printf '%s started=' "$label"
  cat "$rt/started_at.txt" 2>/dev/null || echo pending
  test -f "$rt/exit_code.txt" && { printf '%s_exit=' "$label"; cat "$rt/exit_code.txt"; } || true
done

"$python" - "$control_run/metrics.log" "$method_run/metrics.log" "$pair_rt/paired_resources.csv" <<'PY'
import csv, re, sys
from pathlib import Path

def section(block, start, end):
    a=block.find(start)
    if a<0: return ''
    b=block.find(end, a+len(start))
    return block[a:b if b>=0 else len(block)]

def val(pat, text):
    m=re.search(pat,text)
    return float(m.group(1)) if m else None

def parse(path):
    text=Path(path).read_text(encoding='utf-8',errors='replace') if Path(path).exists() else ''
    ms=list(re.finditer(r'Global Step:\s*(\d+)/480',text))
    rows=[]
    for i,m in enumerate(ms):
        block=text[m.start():ms[i+1].start() if i+1<len(ms) else len(text)]
        env=section(block,' Environment ',' Evaluation ') or section(block,' Environment ',' Replay Buffer ')
        s=val(r'success_once=([-+0-9.eE]+)',env)
        if s is None: continue
        ev=section(block,' Evaluation ',' Replay Buffer ')
        row={'step':int(m.group(1)),'success':s}
        es=val(r'success_once=([-+0-9.eE]+)',ev)
        if es is not None: row['eval']=es
        for key in ('weight_p05','weight_mean','weight_p95','weight_ess_ratio'):
            x=val(rf'actor/rlt_dvac/{key}=([-+0-9.eE]+)',block)
            if x is not None: row[key]=x
        rows.append(row)
    return rows

for label,path in [('control',sys.argv[1]),('method',sys.argv[2])]:
    rows=parse(path)
    if not rows:
        print(label,'NO_COMPLETE_STEP')
        continue
    xs=[r['success']*100 for r in rows]
    r=rows[-1]
    print(label,'step',r['step'],'latest',round(xs[-1],3),'mean',round(sum(xs)/len(xs),3),
          'last5',round(sum(xs[-5:])/len(xs[-5:]),3),'last10',round(sum(xs[-10:])/len(xs[-10:]),3))
    evals=[(r['step'],round(r['eval']*20),round(r['eval']*100,1)) for r in rows if 'eval' in r]
    print(label,'fixed20',evals[-4:])
    if label=='method':
        print(label,'weights',{k:r.get(k) for k in ('weight_p05','weight_mean','weight_p95','weight_ess_ratio')})

rows=[]
with open(sys.argv[3],encoding='utf-8',newline='') as f:
    for r in csv.DictReader(f):
        try: rows.append({k:float(r[k]) for k in ('memory_current','gpu0_mib','gpu1_mib')})
        except Exception: pass
if rows:
    print('resources_latest_gib',round(rows[-1]['memory_current']/2**30,2),round(rows[-1]['gpu0_mib']/1024,2),round(rows[-1]['gpu1_mib']/1024,2))
    print('resources_peak_gib',round(max(r['memory_current'] for r in rows)/2**30,2),round(max(r['gpu0_mib'] for r in rows)/1024,2),round(max(r['gpu1_mib'] for r in rows)/1024,2))
PY

echo PROGRESS
for rt in "$control_rt" "$method_rt"; do
  grep -aE 'Generating Rollout Epochs|Evaluating Rollout Epochs' "$rt/foreground.log" | tail -n 2 || true
done
echo TIMING
for run in "$control_run" "$method_run"; do
  printf '%s\n' "$run"
  grep -aE 'Global Step:|Elapsed:' "$run/metrics.log" 2>/dev/null | tail -n 2 || true
done
echo CHECKPOINTS
for run in "$control_run" "$method_run"; do
  printf '%s ' "$run"
  find "$run" -type d -path '*/checkpoints/global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1
done
echo ERRORS
for rt in "$control_rt" "$method_rt"; do
  printf '%s ' "$rt"
  for p in 'CUDA out of memory' 'OutOfMemoryError' 'WorkerCrashedError' 'ActorDiedError' 'NCCL error'; do
    printf '%s=%s ' "$p" "$(grep -cF "$p" "$rt/foreground.log" 2>/dev/null || true)"
  done
  echo
done
echo CGROUP
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
echo GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
