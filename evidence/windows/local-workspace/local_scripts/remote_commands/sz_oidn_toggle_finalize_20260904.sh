#!/usr/bin/env bash
set -euo pipefail
trial=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-oidn-toggle-20260904
original=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
rl=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
out=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/oidn-toggle-20260904
date -Is
test "$(cat "$out/oidn/exit_code.txt")" = 0
test "$(cat "$out/none/exit_code.txt")" = 0
git -C "$trial" diff --check
git -C "$trial" diff --cached --check
git -C "$trial" diff --stat
git -C "$trial" status --short
sha256sum "$trial/robotwin/envs/vector_env.py" "$original/robotwin/envs/vector_env.py"
git -C "$original" rev-parse HEAD
git -C "$original" status --porcelain
git -C "$rl" rev-parse HEAD
git -C "$rl" status --porcelain
python3 - <<'PY'
import subprocess,urllib.parse,json
from pathlib import Path
p='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-oidn-toggle-20260904'
target=None
for remote in subprocess.check_output(['git','-C',p,'remote'],text=True).splitlines():
 url=subprocess.check_output(['git','-C',p,'remote','get-url','--push',remote],text=True).strip()
 if url.startswith('git@github.com:'): clean=url.split(':',1)[1]
 else:
  u=urllib.parse.urlparse(url)
  clean=u.path.lstrip('/') if u.hostname=='github.com' else ''
 print('REMOTE_TARGET',remote,clean)
 if clean.rstrip('/').removesuffix('.git')=='Yutenji-Nyamu/RoboTwin':target=remote
assert target is not None,'No existing user-owned push remote; do not push official upstream'
print('PUSH_REMOTE',target)
for mode in ['oidn','none']:
 root=Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/oidn-toggle-20260904')/mode
 s=json.loads((root/'summary.json').read_text())
 print('RESULT',mode,len(s['episodes']),sum(x['success'] for x in s['episodes']),len(s['denoiser_calls']))
subprocess.run(['git','-C',p,'add','--','envs/_base_task.py'],check=True)
identity_repo='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo'
name=subprocess.check_output(['git','-C',identity_repo,'config','--get','user.name'],text=True).strip()
email=subprocess.check_output(['git','-C',identity_repo,'config','--get','user.email'],text=True).strip()
assert name=='Yutenji-Nyamu' and email,'Missing established project identity'
subprocess.run(['git','-C',p,'-c',f'user.name={name}','-c',f'user.email={email}','commit','-m','feat(robotwin): expose ray tracing denoiser task option','-m','Keep oidn as the default. Explicit none is set before camera creation. Three inference episodes per mode completed; not a long-run stability claim.'],check=True)
subprocess.run(['timeout','60s','git','-C',p,'push','-u',target,'codex/sz-robotwin-oidn-toggle'],check=True)
PY
git -C "$trial" rev-parse HEAD
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
ps -p 321933,322685,3176215 -o user,pid,etime,stat,comm
