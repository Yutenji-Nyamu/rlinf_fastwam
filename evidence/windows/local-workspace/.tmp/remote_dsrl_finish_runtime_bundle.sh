set -euo pipefail

EXPORT_ROOT=/root/autodl-tmp/experiment_exports/dsrl_pi0_robotwin_formal_v1_20260729
STAGE=$EXPORT_ROOT/runtime_step198
ARCHIVE=$EXPORT_ROOT/dsrl_pi0_robotwin_formal_v1_runtime_step198_20260729.tar.gz

test -d "$STAGE"
test ! -e "$ARCHIVE"

matches=$(grep -R -I -h -o -E \
  'SEETA_SSH_PASSWORD|OPENAI_API_KEY|WANDB_API_KEY|HF_TOKEN|BEGIN (RSA|OPENSSH) PRIVATE KEY' \
  "$STAGE" | sort -u)
files=$(grep -R -I -l -E \
  'SEETA_SSH_PASSWORD|OPENAI_API_KEY|WANDB_API_KEY|HF_TOKEN|BEGIN (RSA|OPENSSH) PRIVATE KEY' \
  "$STAGE" | sort)

test "$matches" = "SEETA_SSH_PASSWORD"
test "$files" = "$STAGE/metadata/dsrl_branch_from_preimplementation.patch"

{
  echo "SECRET_SCAN=PASS_WITH_NAME_ONLY_WHITELIST"
  echo "allowed_literal=SEETA_SSH_PASSWORD"
  echo "allowed_file=metadata/dsrl_branch_from_preimplementation.patch"
  echo "reason=the patch contains process-only credential environment-variable references and redacted placeholders; no credential value is stored"
  echo "all other scanned credential names/private-key headers=absent"
} > "$EXPORT_ROOT/secret_scan_status.txt"

tar -C "$EXPORT_ROOT" -czf "$ARCHIVE" runtime_step198
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

echo "BUNDLE_BUILD=PASS"
echo "EXPORT_ROOT=$EXPORT_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")"
cat "$ARCHIVE.sha256"
du -sh "$STAGE"
