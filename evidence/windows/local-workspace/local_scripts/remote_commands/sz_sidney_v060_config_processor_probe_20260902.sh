set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/data/chenyiteng/cache/huggingface-sidney
venv=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
model=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
"$venv/bin/python" - <<'PY'
from pathlib import Path
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies import make_pre_post_processors

p=Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab')
cfg=PreTrainedConfig.from_pretrained(p, local_files_only=True)
print('CONFIG_OK', type(cfg).__name__)
for name in ['chunk_size','n_action_steps','num_inference_steps','max_state_dim','max_action_dim','dtype','use_relative_actions','action_feature_names','compile_model','device']:
 print(name, getattr(cfg,name,None))
print('inputs', {k:(v.type.value, tuple(v.shape)) for k,v in cfg.input_features.items()})
print('outputs', {k:(v.type.value, tuple(v.shape)) for k,v in cfg.output_features.items()})
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(p))
print('PREPROCESSOR_OK', type(pre).__name__, len(pre.steps))
print('PRE_STEPS', [type(x).__name__ for x in pre.steps])
print('POSTPROCESSOR_OK', type(post).__name__, len(post.steps))
print('POST_STEPS', [type(x).__name__ for x in post.steps])
PY
