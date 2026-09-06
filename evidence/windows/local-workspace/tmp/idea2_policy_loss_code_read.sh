set -eu

repo=/root/autodl-tmp/RLinf_idea2_dvac_train
printf '%s\n' '=== policy loss registry ==='
grep -R -n -E '^def policy_loss|def .*ppo.*loss|logprob_type.*chunk|ratio_clip' \
  "$repo/rlinf/algorithms" | head -n 240

printf '%s\n' '=== registry policy_loss wrapper ==='
file=$(grep -R -l '^def policy_loss' "$repo/rlinf/algorithms" | head -n 1)
printf 'FILE=%s\n' "$file"
line=$(grep -n '^def policy_loss' "$file" | head -n 1 | cut -d: -f1)
start=$((line - 20)); end=$((line + 180))
sed -n "${start},${end}p" "$file"

printf '%s\n' '=== chunk helpers ==='
grep -R -n -E -C 18 'logprob_type.*chunk|chunk.*logprob|sum\(.*-2.*-1|sum\(dim=\(-2, -1\)' \
  "$repo/rlinf/algorithms" | head -n 500

printf '%s\n' '=== PPO actor loss ==='
sed -n '150,312p' "$repo/rlinf/algorithms/losses.py"

printf '%s\n' '=== loss preprocessing ==='
sed -n '270,365p' "$repo/rlinf/algorithms/utils.py"
