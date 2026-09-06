#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
date -Is
id
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits
free -b
df -B1 / /home /data
cat /proc/pressure/memory /proc/pressure/io
python3 - <<'PY'
import os, re, json, datetime, subprocess
from pathlib import Path
runs = {
 'sidney': Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1'),
 'fastwam': Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2'),
}
ansi = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
fatal = re.compile(r'Traceback|CUDA out of memory|RayActorError|NCCL.*error|non.?finite|Segmentation fault|Fatal Python error|OIDN.*[Ee]rror|pthread_key_create|OutOfMemoryError')
for label, run in runs.items():
 print('\nRUN', label, str(run), flush=True)
 for name in ('wrapper.pid','driver.pid','exit_code.txt'):
  p = run/'runtime'/name
  print(name, p.read_text().strip() if p.exists() else 'absent')
  if p.exists() and name.endswith('.pid'):
   pid = p.read_text().strip()
   print(subprocess.run(['ps','-p',pid,'-o','user,pid,ppid,pgid,lstart,etime,stat,%cpu,%mem,args'], capture_output=True,text=True).stdout)
 log = run/'runtime/driver.log'
 if log.exists():
  st = log.stat()
  print('log_stat', json.dumps({'bytes':st.st_size,'mtime':datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat()}))
  matches, steps, evals, tail = [], [], [], []
  count = 0
  with log.open(errors='replace') as f:
   for raw in f:
    line = ansi.sub('',raw).strip()
    if fatal.search(line):
     count += 1
     matches.append(line[-1800:])
     matches=matches[-12:]
    if re.search(r'Global Step:',line): steps.append(line[-2200:]); steps=steps[-14:]
    if re.search(r'eval/|eval_success|eval.*success|success.*eval|Saving checkpoint|Saved checkpoint',line,re.I): evals.append(line[-2400:]); evals=evals[-16:]
    if line: tail.append(line[-1800:]); tail=tail[-20:]
  print('fatal_match_count',count)
  print('fatal_matches',json.dumps(matches))
  print('recent_step_lines',json.dumps(steps))
  print('recent_eval_save_lines',json.dumps(evals))
  print('log_tail',json.dumps(tail))
 print('run_file_inventory')
 for root, dirs, files in os.walk(run):
  rel = Path(root).relative_to(run)
  if len(rel.parts)>=4: dirs[:]=[]
  dirs[:] = [d for d in dirs if d not in ('videos','video','media','wandb')]
  for name in files:
   if name.endswith(('.mp4','.png','.jpg')): continue
   p = Path(root)/name
   st = p.stat()
   print(json.dumps({'path':str(p.relative_to(run)), 'bytes':st.st_size,'mtime':datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat()}))
 print('resource_tail')
 p=run/'runtime/resource.csv'
 if p.exists(): print(subprocess.run(['tail','-n','3',str(p)],capture_output=True,text=True).stdout)
print('\nOWNED_RELEVANT_PROCESSES')
ps=subprocess.run(['ps','-u',str(os.getuid()),'-o','pid,ppid,pgid,etime,stat,%cpu,rss,args'],capture_output=True,text=True).stdout
for line in ps.splitlines():
 if re.search(r'train_embodied|raylet|gcs_server|ray::|run_rl|wrapper|mihomo',line) and not 'python3 -' in line: print(line[:1600])
PY
