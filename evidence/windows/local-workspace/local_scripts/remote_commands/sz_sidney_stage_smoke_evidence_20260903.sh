set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney
P="$ROOT/smokes/parity-core224-m10-compare-existing-v8"
A="$ROOT/smokes/b1-adjust-badseed-retry-m10-phys4-evalrunner-v13r3"
AP="$ROOT/packets/b1-adjust-badseed-retry-m10-phys4-evalrunner-v13r3"
B="$ROOT/smokes/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12"
BP="$ROOT/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12"
OUT="$WT/evidence/smoke_20260903"

test "$(git -C "$WT" rev-parse HEAD)" = bab221afb8bedc32a8f01b171901a347f0258063
test -z "$(git -C "$WT" status --porcelain)"
test ! -e "$OUT"
for dir in "$P" "$A" "$AP" "$B" "$BP"; do test -d "$dir"; done
mkdir -p "$OUT/parity" "$OUT/b1_eval" "$OUT/grpo_step1"

cp -- "$P/report.json" "$OUT/parity/report.json"

for name in command.txt resolved.yaml source-head.txt run.sh run-sha256.txt command-sha256.txt gpu-before.csv memory-before.txt; do
  cp -- "$AP/$name" "$OUT/b1_eval/$name"
done
cp -- "$A/metrics.log" "$OUT/b1_eval/metrics.log"
cp -- "$A/runtime/resources.log" "$OUT/b1_eval/resources.log"
cp -- "$A/runtime/exit_code.txt" "$OUT/b1_eval/exit_code.txt"
cp -- "$A/tensorboard/config.yaml" "$OUT/b1_eval/tensorboard-config.yaml"
A_EVENT=$(find "$A/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' -print -quit)
test -n "$A_EVENT"
cp -- "$A_EVENT" "$OUT/b1_eval/events.out.tfevents"

for name in command.txt resolved.yaml source-head.txt run.sh run-sha256.txt contract.json model-sha256.txt gpu-before.csv memory-before.txt; do
  cp -- "$BP/$name" "$OUT/grpo_step1/$name"
done
cp -- "$B/metrics.log" "$OUT/grpo_step1/metrics.log"
cp -- "$B/runtime/resources.log" "$OUT/grpo_step1/resources.log"
cp -- "$B/runtime/exit_code.txt" "$OUT/grpo_step1/exit_code.txt"
cp -- "$B/tensorboard/config.yaml" "$OUT/grpo_step1/tensorboard-config.yaml"
B_EVENT=$(find "$B/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' -print -quit)
test -n "$B_EVENT"
cp -- "$B_EVENT" "$OUT/grpo_step1/events.out.tfevents"

grep -E 'Using flexible placement|Loaded norm stats|reset error with seed|Evaluating Rollout Epochs|eval/reward|Global Step:|episode_len=|return=|success_' \
  "$A/runtime/driver.log" > "$OUT/b1_eval/driver.key.log"
grep -E 'Using flexible placement|Loaded norm stats|Generating Rollout Epochs|Evaluating Rollout Epochs|Saving checkpoint|Global Step:|num_trajectories=|return=|success_|advantages_|approx_kl=|clip_fraction=|grad_norm=|policy_loss=|total_loss=' \
  "$B/runtime/driver.log" > "$OUT/grpo_step1/driver.key.log"

for pair in "$A:$OUT/b1_eval" "$B:$OUT/grpo_step1"; do
  SRC=${pair%%:*}
  DST=${pair#*:}
  awk '
    BEGIN {print "timestamp,gpu,memory_used_mib,memory_total_mib,utilization_percent,mem_available_kb"}
    /^[0-9][0-9][0-9][0-9]-/ {ts=$0; n=0; next}
    /^[0-9]+,/ {n++; line=$0; gsub(/, /, ",", line); split(line,a,","); gpu[n]=a[1]; used[n]=a[2]; total[n]=a[3]; util[n]=a[4]; next}
    /^MemAvailable:/ {for(i=1;i<=n;i++) print ts "," gpu[i] "," used[i] "," total[i] "," util[i] "," $2}
  ' "$SRC/runtime/resources.log" > "$DST/resources.csv"
done

CKPT="$B/pi05_sidney_move_grpo_smoke1_2gpu64x1_g8_b512_u2_m10_fixed8_phys45_localshard_v12/checkpoints/global_step_1"
test -d "$CKPT"
find "$CKPT" -type f -printf '%P\t%s bytes\n' | sort > "$OUT/grpo_step1/checkpoint-layout.txt"
printf '%s\n' "$CKPT" > "$OUT/grpo_step1/checkpoint-server-path.txt"

find "$OUT" -type f -size +1M -print -quit | grep -q . && { echo 'unexpected file >1MiB' >&2; exit 21; } || true
du -sb "$OUT"
find "$OUT" -type f -printf '%s %P\n' | sort -n
echo SIDNEY_EVIDENCE_FILES_STAGED
