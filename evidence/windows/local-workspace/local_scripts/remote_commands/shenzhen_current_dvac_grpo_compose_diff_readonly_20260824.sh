#!/usr/bin/env bash
set -euo pipefail
OUT=/data/chenyiteng/results/rlinf-shenzhen/grpo/pretest-current-dvac-global-z-20260824-v2
printf 'MARKER=SZ_CURRENT_DVAC_GRPO_COMPOSE_DIFF_V1\n'
diff -u "$OUT/base.resolved.yaml" "$OUT/dvac.resolved.yaml" || true
grep -A14 -B2 'dvac_gradient_weighting' "$OUT/base.resolved.yaml"
grep -A14 -B2 'dvac_gradient_weighting' "$OUT/dvac.resolved.yaml"
printf 'MARKER=SZ_CURRENT_DVAC_GRPO_COMPOSE_DIFF_OK\n'
