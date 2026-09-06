set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
venv=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
model=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
"$venv/bin/python" - <<'PY'
from pathlib import Path
from safetensors.torch import load_file
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

p=Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab')
cfg=PreTrainedConfig.from_pretrained(p, local_files_only=True)
cfg.device='cpu'
policy=PI05Policy(cfg)
original=load_file(str(p/'model.safetensors'), device='cpu')
fixed=policy._fix_pytorch_state_dict_keys(original, cfg)
remapped={(k if k.startswith('model.') else f'model.{k}'):v for k,v in fixed.items()}
result=policy.load_state_dict(remapped, strict=True)
print('STRICT_LOAD_OK')
print('checkpoint_keys',len(original),'remapped_keys',len(remapped),'model_keys',len(policy.state_dict()))
print('missing',list(result.missing_keys),'unexpected',list(result.unexpected_keys))
print('params',sum(x.numel() for x in policy.parameters()))
PY
