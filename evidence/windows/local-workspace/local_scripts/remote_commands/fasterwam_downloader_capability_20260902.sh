set -euo pipefail

for name in aria2c axel wget curl; do
  printf '%s ' "$name"
  command -v "$name" || true
done
