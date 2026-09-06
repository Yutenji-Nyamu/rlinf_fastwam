#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
date -Is
python3 - <<'PY'
from pathlib import Path
root=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo')
p=root/'rlinf/workers/actor/embodied_fsdp_actor_worker.py'
if not p.exists():
 candidates=list((root/'rlinf/workers/actor').glob('*fsdp*.py'));print('CANDIDATES',candidates)
else:
 lines=p.read_text().splitlines()
 hits={j for i,l in enumerate(lines) if any(k in l for k in ['global_batch_size','update_epoch','optimizer.step','num_steps','gradient_accumulation']) for j in range(max(0,i-3),min(len(lines),i+6))}
 print('ACTOR_SOURCE',p)
 for i in sorted(hits):print(f'{i+1}: {lines[i]}')
rt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-clean-oidn-off-20260904')
for rel,terms in [('rlinf/envs/robotwin/robotwin_env.py',['task_config','VectorEnv']),('robotwin/envs/vector_env.py',['setup_demo','args'])]:
 p=(root if rel.startswith('rlinf/') else rt)/rel
 lines=p.read_text().splitlines()
 hits={j for i,l in enumerate(lines) if any(k in l for k in terms) for j in range(max(0,i-2),min(len(lines),i+4))}
 print('CONFIG_CHAIN',p)
 for i in sorted(hits):print(f'{i+1}: {lines[i]}')
PY
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1
tail -n 15 "$RUN/runtime/driver.log"
date -Is
