set -euo pipefail
CKPT=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
BASE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PY=/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/python

echo '=== key accounting details ==='
"$PY" - <<'PY'
from collections import Counter
from pathlib import Path
from safetensors import safe_open

p=Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab/model.safetensors')
with safe_open(str(p), framework='pt', device='cpu') as f:
    ks=list(f.keys())
print('source_count',len(ks))
print('top1', Counter(k.split('.')[0] for k in ks))
print('top3', Counter('.'.join(k.split('.')[:3]) for k in ks).most_common(20))
for needle in ('action_time','time_mlp','embed_tokens','lm_head','language_model','action_in_proj','action_out_proj'):
    hits=[k for k in ks if needle in k]
    print('NEEDLE',needle,'COUNT',len(hits))
    for k in hits[:12]: print(' ',k)
PY

echo '=== norm exact comparison and relevant values ==='
"$PY" - <<'PY'
from pathlib import Path
import torch
from safetensors.torch import load_file

r=Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab')
pre=load_file(str(r/'policy_preprocessor_step_3_normalizer_processor.safetensors'))
post=load_file(str(r/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
print('pre_keys',len(pre),'post_keys',len(post),'same_keyset',set(pre)==set(post))
bad=[]
for k in sorted(set(pre)&set(post)):
    if not torch.equal(pre[k],post[k]): bad.append(k)
print('nonidentical_pre_post',bad)
for field in ('observation.state','action'):
    for stat in ('mean','std','min','max','count'):
        k=f'{field}.{stat}'
        print(k,pre[k].tolist())
PY

echo '=== base converter full ==='
nl -ba "$BASE/rlinf/utils/ckpt_convertor/openpi/openpi_pytorch_to_openpi_rlinf.py"
echo '=== base convert entry full ==='
nl -ba "$BASE/rlinf/utils/ckpt_convertor/openpi/convert.py"

echo '=== implementation diff now (read-only) ==='
git -C "$WT" status --short --branch
git -C "$WT" diff --stat
git -C "$WT" diff -- rlinf/utils/ckpt_convertor/openpi/openpi_pytorch_to_openpi_rlinf.py rlinf/utils/ckpt_convertor/openpi/convert.py

echo '=== possible existing pi05 native dirs ==='
find /data/chenyiteng -maxdepth 7 -type d \( -iname '*Pi05*RoboTwin*SFT*' -o -iname '*pi05*adjust*bottle*' -o -iname 'assets' \) 2>/dev/null | head -100
