set -eu

RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1

echo '=== RUN FILES ==='
find "$RUN" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 120

echo '=== FAILURE PATTERN FILES ==='
grep -RIlE 'OIDN|pthread_key_create|PyGILState_Release|invalid handle|NCCL|TCPStore|ActorDiedError|RayActorError|SYSTEM_ERROR|WorkerCrashedError' "$RUN" 2>/dev/null | sort || true

echo '=== RAY SESSION ==='
readlink -f /tmp/ray/session_latest || true
find /tmp/ray/session_latest/logs -maxdepth 1 -type f \( -name '*3590591*' -o -name '*3590594*' -o -name '*3590534*' -o -name '*3589674*' -o -name 'raylet*' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort || true

echo '=== CONFIG OFFLOAD/EVAL ==='
grep -nE 'enable_offload|num_envs|rollout_epoch|eval_interval|save_interval|video' "$RUN/resolved.yaml" 2>/dev/null | head -n 120 || true
