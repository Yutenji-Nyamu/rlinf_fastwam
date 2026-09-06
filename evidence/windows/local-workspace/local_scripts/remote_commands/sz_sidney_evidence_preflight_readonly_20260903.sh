set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney
P="$ROOT/smokes/parity-core224-m10-compare-existing-v8"
A="$ROOT/smokes/b1-adjust-badseed-retry-m10-phys4-evalrunner-v13r3"
AP="$ROOT/packets/b1-adjust-badseed-retry-m10-phys4-evalrunner-v13r3"
B="$ROOT/smokes/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12"
BP="$ROOT/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12"
printf '%s\n' '--- git ---'
git -C "$WT" branch --show-current
git -C "$WT" rev-parse HEAD
git -C "$WT" status --short
git -C "$WT" remote -v | sed -E 's#(https?://)[^/@]+@#\1<redacted>@#'
printf '%s\n' '--- candidate files ---'
for f in \
  "$P/report.json" \
  "$A/metrics.log" "$A/runtime/driver.log" "$A/runtime/resources.log" "$A/runtime/exit_code.txt" \
  "$A/tensorboard/config.yaml" \
  "$AP/command.txt" "$AP/resolved.yaml" "$AP/source-head.txt" "$AP/run.sh" "$AP/run-sha256.txt" "$AP/command-sha256.txt" "$AP/gpu-before.csv" "$AP/memory-before.txt" \
  "$B/metrics.log" "$B/runtime/driver.log" "$B/runtime/resources.log" "$B/runtime/exit_code.txt" \
  "$B/tensorboard/config.yaml" \
  "$BP/command.txt" "$BP/resolved.yaml" "$BP/source-head.txt" "$BP/run.sh" "$BP/run-sha256.txt" "$BP/contract.json" "$BP/model-sha256.txt" "$BP/gpu-before.csv" "$BP/memory-before.txt"; do
  if test -f "$f"; then stat -c '%s %n' "$f"; else printf 'MISSING %s\n' "$f"; fi
done
printf '%s\n' '--- tensorboard files ---'
find "$A/tensorboard" "$B/tensorboard" -maxdepth 1 -type f -printf '%s %p\n' 2>/dev/null | sort
printf '%s\n' '--- credential-pattern scan small text ---'
grep -ERin --binary-files=without-match '(password[[:space:]]*[:=]|api[_-]?key[[:space:]]*[:=]|authorization:[[:space:]]*bearer|hf_[A-Za-z0-9]{20,}|github_pat_)' \
  "$P/report.json" "$A/metrics.log" "$A/runtime/driver.log" "$AP/command.txt" "$AP/resolved.yaml" \
  "$B/metrics.log" "$B/runtime/driver.log" "$BP/command.txt" "$BP/resolved.yaml" || true
