#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
sed -n '230,340p' "$root/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"
sed -n '20,145p' "$root/rlinf/hybrid_engines/fsdp/strategy/fsdp.py"
sed -n '170,225p' "$root/rlinf/hybrid_engines/fsdp/strategy/fsdp.py"
grep -RnA 14 'def torch_dtype_from_precision' "$root/rlinf/utils" --include='*.py' || true
grep -nA 40 'def model_provider_func' "$root/rlinf/workers/actor/embodied_fsdp_actor_worker.py" "$root/rlinf/workers/actor/fsdp_dagger_policy_worker.py"
cat "$root/examples/embodiment/config/hybrid_engines/fsdp.yaml"
