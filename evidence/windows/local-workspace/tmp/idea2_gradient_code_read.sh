set -eu

repo=/root/autodl-tmp/RLinf_idea2_dvac_train
weighting=$(find "$repo" -type f -name 'dvac_train_weighting.py' | head -n 1)
actor="$repo/rlinf/workers/actor/fsdp_actor_worker.py"
manager="$repo/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"
strategy="$repo/rlinf/hybrid_engines/fsdp/strategy/fsdp.py"

printf 'WEIGHTING=%s\n' "$weighting"
printf '%s\n' '=== weighting implementation ==='
sed -n '1,360p' "$weighting"

printf '%s\n' '=== actor imports/config ==='
sed -n '1,75p' "$actor"
printf '%s\n' '=== actor new loss path ==='
sed -n '1615,1725p' "$actor"
printf '%s\n' '=== actor update/grad clip calls ==='
grep -n -E -C 16 'clip_grad_norm_|optimizer.step|backward' "$actor" | tail -n 180

printf '%s\n' '=== model manager clip ==='
sed -n '405,440p' "$manager"
printf '%s\n' '=== fsdp strategy clip ==='
sed -n '340,480p' "$strategy"

printf '%s\n' '=== source identity ==='
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
