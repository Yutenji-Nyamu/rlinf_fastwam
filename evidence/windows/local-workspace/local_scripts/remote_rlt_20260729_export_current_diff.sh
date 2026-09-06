set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
OUTPUT_PATH=/root/autodl-tmp/experiment_exports/rlt_pre_smoke_20260729/server_current_before_final_review.patch
temp_index=$(mktemp)
trap 'rm -f "$temp_index"' EXIT
rm -f "$temp_index"

cd "$RLT_ROOT"
GIT_INDEX_FILE="$temp_index" git read-tree HEAD
GIT_INDEX_FILE="$temp_index" git add -A
GIT_INDEX_FILE="$temp_index" git diff \
  --cached \
  --binary \
  --no-ext-diff \
  --output="$OUTPUT_PATH" \
  HEAD

stat --format='%s' "$OUTPUT_PATH"
sha256sum "$OUTPUT_PATH"
