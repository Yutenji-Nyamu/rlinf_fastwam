set -eu

RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
RAY=/data/chenyiteng/ray/rlt-dsrl-v3/session_2026-08-23_16-27-53_911161_321906/logs

echo '=== DRIVER HASH ==='
sha256sum "$RUN/runtime/driver.log"

echo '=== MATCHING WORKER FILES ==='
find "$RAY" -maxdepth 2 -type f \( -name '*3590591*' -o -name '*3590594*' -o -name '*3590534*' -o -name '*3589674*' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort || true

echo '=== RAYLET FAILURE REFERENCES ==='
grep -nE '3590591|3590594|c3f95d5f2c8d423bb1ce0c6162010000' "$RAY/raylet.out" "$RAY/raylet.err" 2>/dev/null | tail -n 100 || true

echo '=== CORE-WORKER FAILURE REFERENCES ==='
for f in $(find "$RAY" -maxdepth 2 -type f \( -name '*3590591*' -o -name '*3590594*' -o -name '*3590534*' \) 2>/dev/null); do
  m=$(grep -nEi 'OIDN|pthread_key_create|PyGILState|invalid handle|SIGABRT|SIGSEGV|SYSTEM_ERROR|WorkerCrashed|TCPStore|NCCL|unavailable|socket closed|exit' "$f" 2>/dev/null | tail -n 80 || true)
  if [ -n "$m" ]; then
    echo "--- $f"
    printf '%s\n' "$m"
  fi
done

echo '=== RESOLVED OFFLOAD/EVAL ==='
grep -nE 'enable_offload|num_envs|rollout_epoch|eval_interval|save_interval|video' "$RUN/runtime/resolved.yaml" 2>/dev/null | head -n 160 || true
