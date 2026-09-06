set -eu
echo '[identity]'
hostname
pwd
id -u
echo '[repo]'
cd /root/autodl-tmp/RLinf_fastwam_rlinf
git rev-parse HEAD
git status --short --branch
echo '[files]'
sha256sum \
  rlinf/models/embodiment/fastwam/builder.py \
  rlinf/models/embodiment/fastwam/fastwam_policy.py \
  rlinf/models/embodiment/fastwam/fastwam_rl.py \
  rlinf/workers/actor/fsdp_actor_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/algorithms/advantages.py \
  rlinf/algorithms/losses.py \
  rlinf/hybrid_engines/fsdp/fsdp_model_manager.py \
  rlinf/models/embodiment/modules/value_head.py
echo '[ppo-related configs]'
find examples/embodiment/config -maxdepth 2 -type f \( -iname '*ppo*fastwam*' -o -iname '*ppo*openpi*' -o -iname '*ppo*motus*' \) -print | sort
