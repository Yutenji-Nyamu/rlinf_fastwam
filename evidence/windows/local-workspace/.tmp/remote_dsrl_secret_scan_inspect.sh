set -euo pipefail

STAGE=/root/autodl-tmp/experiment_exports/dsrl_pi0_robotwin_formal_v1_20260729/runtime_step198

grep -R -I -h -o -E \
  'SEETA_SSH_PASSWORD|OPENAI_API_KEY|WANDB_API_KEY|HF_TOKEN|BEGIN (RSA|OPENSSH) PRIVATE KEY' \
  "$STAGE" | sort | uniq -c

echo "FILES_BEGIN"
grep -R -I -l -E \
  'SEETA_SSH_PASSWORD|OPENAI_API_KEY|WANDB_API_KEY|HF_TOKEN|BEGIN (RSA|OPENSSH) PRIVATE KEY' \
  "$STAGE" | sort
echo "FILES_END"
