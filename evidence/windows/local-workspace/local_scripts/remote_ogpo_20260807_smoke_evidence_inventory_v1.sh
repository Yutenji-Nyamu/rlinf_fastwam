set -euo pipefail

RUN_ROOT=/root/autodl-tmp/experiments/ogpo_robotwin_smoke_20260807_v1
RUNTIME_ROOT=/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime

test -d "$RUN_ROOT"
test -d "$RUNTIME_ROOT"

echo '=== runtime files: bytes path sha256 ==='
while IFS= read -r -d '' path; do
  printf '%s\t%s\t' "$(stat -c %s "$path")" "$path"
  sha256sum "$path" | awk '{print $1}'
done < <(find "$RUNTIME_ROOT" -maxdepth 1 -type f -print0 | sort -z)

echo '=== small run evidence: bytes path sha256 ==='
while IFS= read -r -d '' path; do
  size=$(stat -c %s "$path")
  if test "$size" -le 104857600; then
    printf '%s\t%s\t' "$size" "$path"
    sha256sum "$path" | awk '{print $1}'
  fi
done < <(find "$RUN_ROOT" -type f \( -name 'metrics.log' -o -name 'config.yaml' -o -name 'events.out.tfevents.*' -o -name 'complete.json' \) -print0 | sort -z)

echo '=== checkpoint bytes ==='
find "$RUN_ROOT" -type f -path '*/checkpoints/*' -printf '%s\t%p\n' | sort -n
echo 'OGPO_ROBOTWIN_SMOKE_EVIDENCE_INVENTORY_OK'
