set -eu
sudo -S -p '' bash -c '
for f in \
  /home/guorenjie/research/smolvla-libero-clp/scripts/smoke_pi0_train_vram.sh \
  /home/guorenjie/research/smolvla-libero-clp/scripts/smoke_pi0_vlm_lora.sh; do
  echo "=== $f"
  grep -n -B6 -A4 "rm -rf" "$f" || true
done
'
