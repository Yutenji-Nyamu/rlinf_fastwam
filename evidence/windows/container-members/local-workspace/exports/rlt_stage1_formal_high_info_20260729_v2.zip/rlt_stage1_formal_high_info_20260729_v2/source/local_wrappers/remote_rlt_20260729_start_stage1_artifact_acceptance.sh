#!/usr/bin/env bash
set -euo pipefail

RUN_SCRIPT=/root/autodl-tmp/tmp/rlt_stage1_artifact_acceptance_20260729_v2.sh
OUTPUT_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2

if pgrep -af 'validate_robotwin_rlt_stage1_artifact.py' | grep -v pgrep; then
  echo "an artifact validation process is already running" >&2
  exit 91
fi
mkdir -p "${OUTPUT_ROOT}"
nohup bash "${RUN_SCRIPT}" >"${OUTPUT_ROOT}/launcher.log" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${OUTPUT_ROOT}/launcher.pid"
echo "pid=${pid}"
echo "output=${OUTPUT_ROOT}"
