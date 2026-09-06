set -eu
sudo -S -p '' bash -c '
file=/home/guorenjie/research/smolvla-libero-clp/lerobot/src/lerobot/policies/pi0/clp_prune_pi0.py
sed -n "1,260p" "$file"
echo "=== HOOKS ==="
grep -n -B6 -A14 "clp_remove_indices" \
  /home/guorenjie/research/smolvla-libero-clp/lerobot/src/lerobot/policies/pi0/configuration_pi0.py \
  /home/guorenjie/research/smolvla-libero-clp/lerobot/src/lerobot/policies/pi0/modeling_pi0.py
'
