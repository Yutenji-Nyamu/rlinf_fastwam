#!/usr/bin/env bash
set -o pipefail
cd /root/autodl-tmp/RLinf_rlt_dvac_success_bc
timeout 180 env \
  PYTHONPATH=/root/autodl-tmp/RLinf_rlt_dvac_success_bc:/root/autodl-tmp/RoboTwin_RLinf \
  OMP_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  /root/autodl-tmp/RLinf/.venv/bin/python -X importtime \
  -c 'import rlinf.workers.actor.fsdp_rlt_ac_policy_worker; print("IMPORT_OK")' \
  2>&1 | tail -n 180
