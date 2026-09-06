set -eu
run=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval16x2-gpu6-20260905-v7
date -Is
grep -c 'Forcing gradient checkpointing to be enabled for Gemma expert model' "$run/driver.log" || true
grep -E 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:|Saving checkpoint' "$run/driver.log" | tail -n 8
find "$run/success_data" -type f -printf '%f %s bytes\n'
tail -n 1 "$run/resource.csv"
