set -euo pipefail
source /etc/profile.d/mihomo-proxy.sh
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=4
model=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
venv=/home/chenyiteng/venvs/lerobot-v043-sidney-py310
"$venv/bin/python" - "$model" <<'PY'
import json, sys, traceback
from pathlib import Path
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

p=Path(sys.argv[1])
raw=json.loads((p/'config.json').read_text())
print('RAW', {k: raw.get(k) for k in ['type','chunk_size','n_action_steps','num_inference_steps','max_state_dim','max_action_dim','use_relative_actions','normalization_mapping','dtype','compile_model']})
cfg=PreTrainedConfig.from_pretrained(p)
print('CONFIG_TYPE',type(cfg).__name__)
for k in ['chunk_size','n_action_steps','num_inference_steps','max_state_dim','max_action_dim','dtype','compile_model']:
 print('CFG',k,getattr(cfg,k,None))
print('CFG_HAS_RELATIVE',hasattr(cfg,'use_relative_actions'),getattr(cfg,'use_relative_actions',None))
print('CFG_NORM',cfg.normalization_mapping)
assert cfg.chunk_size==50 and cfg.n_action_steps==50 and cfg.num_inference_steps==10
assert cfg.max_state_dim==32 and cfg.max_action_dim==32
cfg.device='cuda'
pre,post=make_pre_post_processors(
 cfg,
 pretrained_path=p,
 preprocessor_overrides={'device_processor': {'device':'cuda'}},
)
print('PRE', [type(x).__name__ for x in pre.steps])
print('POST', [type(x).__name__ for x in post.steps])
for pipe,name in [(pre,'pre'),(post,'post')]:
 for step in pipe.steps:
  if 'Normalizer' in type(step).__name__:
   stats=getattr(step,'stats',None)
   print('STATS',name,type(step).__name__,None if stats is None else {k:{kk:tuple(vv.shape) for kk,vv in sv.items()} for k,sv in stats.items()})
print('PROCESSOR_OK')
PY
