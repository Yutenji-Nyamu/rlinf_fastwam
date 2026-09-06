set -u

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
WORKER="$RLT_ROOT/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py"

grep -n -A 150 -B 10 'def _rlt_resume_contract' "$WORKER"
printf '%s\n' 'TEST_REFERENCES'
grep -R -n -E \
  'bootstrap_type.*resume|resume.*bootstrap_type|rlt_resume_contract' \
  "$RLT_ROOT/tests" "$RLT_ROOT/rlinf" \
  --include='*.py' || true
