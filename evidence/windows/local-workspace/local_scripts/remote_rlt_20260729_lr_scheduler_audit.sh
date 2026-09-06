set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python

printf '%s\n' 'TIME'
date -Is
printf '%s\n' 'GIT'
git -C "$RLT_ROOT" branch --show-current
git -C "$RLT_ROOT" rev-parse HEAD
git -C "$RLT_ROOT" diff --exit-code -- \
  examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
printf '%s\n' 'SOURCE_REFERENCES'
grep -R -n -E \
  'def build_optimizer|optimizer\.defaults|get_scheduler|min_lr_rate|min_lr' \
  "$RLT_ROOT/rlinf" "$RLT_ROOT/examples/sft" \
  --include='*.py' --include='*.yaml' | head -220
printf '%s\n' 'FSDP_MODEL_MANAGER'
sed -n '450,565p' \
  "$RLT_ROOT/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"
printf '%s\n' 'FSDP_SCHEDULER_UTILITY'
sed -n '520,565p' \
  "$RLT_ROOT/rlinf/hybrid_engines/fsdp/utils.py"
printf '%s\n' 'TRANSFORMERS_VERSION_AND_FACTORY'
"$PYTHON_BIN" -B - <<'PY'
import inspect
import transformers
from transformers import optimization

print("transformers", transformers.__version__)
print(inspect.getsource(optimization.get_cosine_with_min_lr_schedule_with_warmup))
PY
