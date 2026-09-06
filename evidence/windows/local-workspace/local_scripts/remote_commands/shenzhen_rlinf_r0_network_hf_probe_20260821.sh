#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
printf 'https_proxy=%s\n' "${https_proxy:-${HTTPS_PROXY:-unset}}"

printf 'rlinf_raw_http='
curl --fail --silent --show-error --location --max-time 30 \
  --output /dev/null --write-out '%{http_code}\n' \
  https://raw.githubusercontent.com/RLinf/RLinf/7d07a4212ee6858cc333e1d4fab7a37256d1f839/README.md
printf 'pi0_revision_http='
curl --fail --silent --show-error --location --max-time 30 \
  --output /dev/null --write-out '%{http_code}\n' \
  https://huggingface.co/api/models/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/revision/92684e50dca1a5f75adc8d332046c4cf4fa7a3d0
printf 'tokenizer_revision_http='
curl --fail --silent --show-error --location --max-time 30 \
  --output /dev/null --write-out '%{http_code}\n' \
  https://huggingface.co/api/models/RLinf/openpi_tokenizer/revision/befaa248e4f82954b625a421658f933dfd1a97a0
