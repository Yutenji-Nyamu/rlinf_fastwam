set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
temp_index=$(mktemp)
trap 'rm -f "$temp_index"' EXIT
rm -f "$temp_index"

cd "$RLT_ROOT"
GIT_INDEX_FILE="$temp_index" git read-tree HEAD
GIT_INDEX_FILE="$temp_index" git add -A

printf '%s\n' '--- FULL DIFF CHECK ---'
GIT_INDEX_FILE="$temp_index" git diff --cached --check HEAD
printf '%s\n' '--- FULL STATUS ---'
git status --short
printf '%s\n' '--- FULL NAME STATUS ---'
GIT_INDEX_FILE="$temp_index" git diff --cached --name-status HEAD
printf '%s\n' '--- FULL STAT ---'
GIT_INDEX_FILE="$temp_index" git diff --cached --stat HEAD
printf '%s\n' '--- NEW FILE SIZE GUARD ---'
while IFS= read -r path; do
  if [ -f "$path" ]; then
    size=$(stat --format='%s' "$path")
    printf '%s\t%s\n' "$size" "$path"
    if [ "$size" -gt 1048576 ]; then
      printf 'unexpected file larger than 1 MiB: %s\n' "$path" >&2
      exit 1
    fi
  fi
done <<EOF
$(GIT_INDEX_FILE="$temp_index" git diff --cached --name-only --diff-filter=A HEAD)
EOF

if GIT_INDEX_FILE="$temp_index" git diff --cached --no-ext-diff HEAD |
  grep -Eq \
    'BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|SEETA_SSH_PASSWORD[[:space:]]*='; then
  printf '%s\n' 'potential credential pattern detected in full diff' >&2
  exit 1
fi
printf '%s\n' 'CREDENTIAL_PATTERN_SCAN_OK'

printf '%s\n' '--- REMOTES ---'
git remote -v
printf '%s\n' '--- BRANCH ---'
git branch -vv
