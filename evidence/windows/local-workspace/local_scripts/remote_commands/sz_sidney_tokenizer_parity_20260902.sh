set -euo pipefail
tmp=$(mktemp /data/chenyiteng/tmp/sidney-token-parity-XXXXXX.json)
trap 'rm -f "$tmp"' EXIT
export HF_HOME=/data/chenyiteng/cache/huggingface-sidney
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
lrpy=/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/python
opipy=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
model=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
CUDA_VISIBLE_DEVICES=4 "$lrpy" - "$tmp" <<'PY'
import json, sys, torch
from pathlib import Path
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

model=Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab')
cfg=PreTrainedConfig.from_pretrained(model, local_files_only=True)
pre,_=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(model))
rows=[]
for task in ['adjust_bottle','move_stapler_pad']:
 raw={
  OBS_STATE: torch.zeros(14,dtype=torch.float32),
  'observation.images.cam_high': torch.zeros((480,640,3),dtype=torch.uint8),
  'observation.images.cam_left_wrist': torch.zeros((480,640,3),dtype=torch.uint8),
  'observation.images.cam_right_wrist': torch.zeros((480,640,3),dtype=torch.uint8),
  'task': task,
 }
 out=pre(raw)
 rows.append({
  'task':task,
  'normalized_state':out[OBS_STATE][0,:14].detach().cpu().tolist(),
  'tokens':out[OBS_LANGUAGE_TOKENS][0].detach().cpu().tolist(),
  'mask':[bool(x) for x in out[OBS_LANGUAGE_ATTENTION_MASK][0].detach().cpu().tolist()],
 })
with open(sys.argv[1],'w',encoding='utf-8') as f: json.dump(rows,f)
print('LEROBOT_TOKENS_READY',[(r['task'],sum(r['mask'])) for r in rows])
PY
"$opipy" - "$tmp" <<'PY'
import json,sys,numpy as np
from openpi.models.tokenizer import PaligemmaTokenizer
rows=json.load(open(sys.argv[1],encoding='utf-8'))
tok=PaligemmaTokenizer(max_len=200)
for row in rows:
 tokens,mask=tok.tokenize(row['task'],np.asarray(row['normalized_state'],dtype=np.float32))
 lt=np.asarray(row['tokens'])
 lm=np.asarray(row['mask'],dtype=bool)
 print(row['task'],'token_equal',bool(np.array_equal(lt,tokens)),'mask_equal',bool(np.array_equal(lm,mask)),'active',int(mask.sum()))
 if not np.array_equal(lt,tokens):
  idx=np.flatnonzero(lt!=tokens)
  print('token_diff_first',idx[:10].tolist(),[(int(lt[i]),int(tokens[i])) for i in idx[:10]])
 if not np.array_equal(lm,mask):
  idx=np.flatnonzero(lm!=mask)
  print('mask_diff_first',idx[:10].tolist())
 assert np.array_equal(lt,tokens)
 assert np.array_equal(lm,mask)
print('TOKEN_PARITY_OK')
PY
