#!/usr/bin/env bash
set -euo pipefail
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_resume250_to480_20260730_v1/runtime
printf 'PREFLIGHT_SHA256\t'
sha256sum "$runtime/stage1_binding_preflight.json" | awk '{print $1}'
printf 'PREFLIGHT_JSON_BEGIN\n'
cat "$runtime/stage1_binding_preflight.json"
printf 'PREFLIGHT_JSON_END\n'
printf 'PREFLIGHT_STDOUT_BEGIN\n'
cat "$runtime/stage1_binding_preflight.stdout"
printf 'PREFLIGHT_STDOUT_END\n'
