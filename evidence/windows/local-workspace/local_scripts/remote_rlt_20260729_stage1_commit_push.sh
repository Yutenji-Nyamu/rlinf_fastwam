set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
EXPECTED=$(mktemp /root/autodl-tmp/tmp/rlt_stage1_commit_expected.XXXXXX)
ACTUAL=$(mktemp /root/autodl-tmp/tmp/rlt_stage1_commit_actual.XXXXXX)

cleanup() {
  rm -f -- "$EXPECTED" "$ACTUAL"
}
trap cleanup EXIT

cd "$RLT_ROOT"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = e4127fd49e38362161eac08c551a7a98c11e9802
git diff --cached --quiet

cat > "$EXPECTED" <<'EOF'
HANDOFF.md
examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md
docs/rlinf-robotwin-pi0-rltoken/02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md
docs/rlinf-robotwin-pi0-rltoken/evidence/DISK_AUDIT_COMMANDS_20260729.md
docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/README.md
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/exact_commands.txt
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/exact_commands_addendum.md
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/lr_scheduler_contract.json
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1a_driver.log
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1a_reload_driver.log
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1a_reload_resources.csv
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1a_resources.csv
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1b_driver.log
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1b_resources.csv
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/s1a_resolved.yaml
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/s1b_resolved.yaml
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/source_configs/robotwin_rlt_stage1_sft_openpi.yaml
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/source_configs/robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke.yaml
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/stage1_postcheck.json
EOF
sort -o "$EXPECTED" "$EXPECTED"

while IFS= read -r path; do
  git add -f -- "$path"
done < "$EXPECTED"

git diff --cached --name-only | sort > "$ACTUAL"
diff -u "$EXPECTED" "$ACTUAL"
git diff --cached --check
test -z "$(git diff --name-only)"
test -z "$(git ls-files --others --exclude-standard)"

git commit -m "fix(rlt): validate Stage 1 smoke scheduler"
main_commit=$(git rev-parse HEAD)

timeout --signal=TERM --kill-after=10s 90s \
  git push personal codex/rlt-pi0-robotwin

printf 'MAIN_COMMIT=%s\n' "$main_commit"
printf '%s\n' 'STATUS'
git status --short
printf '%s\n' 'AHEAD_BEHIND'
git rev-list --left-right --count HEAD...@{upstream}
printf '%s\n' 'REMOTE_HEAD'
git ls-remote personal refs/heads/codex/rlt-pi0-robotwin
