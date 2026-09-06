set -eu
cd /root/autodl-tmp/RLinf_fastwam_rlinf
git diff -- rlinf/hybrid_engines/fsdp/strategy/fsdp2.py rlinf/models/__init__.py rlinf/workers/rollout/hf/huggingface_worker.py
sed -n '1,240p' examples/embodiment/config/env/robotwin_adjust_bottle.yaml
rg -n 'use_fixed_reset_state_ids|reset_state_ids|group_size|seed' rlinf/envs/robotwin /root/autodl-tmp/RoboTwin_RLinf -g '*.py' -g '*.yaml' | head -n 240
