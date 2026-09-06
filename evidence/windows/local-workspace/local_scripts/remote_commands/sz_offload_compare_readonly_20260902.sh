set -u
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
FAST=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2/runtime/resolved.yaml
PI05=/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2/runtime/resolved.yaml
"$PY" - "$FAST" "$PI05" <<'PY'
import sys, yaml
for name, path in zip(('fastwam256','pi05'), sys.argv[1:]):
    with open(path, encoding='utf-8') as f:
        cfg=yaml.safe_load(f)
    print(name, path)
    for split in ('train','eval'):
        e=cfg['env'][split]
        print(' env.'+split,
              'total_num_envs='+str(e.get('total_num_envs')),
              'rollout_epoch='+str(e.get('rollout_epoch')),
              'enable_offload='+str(e.get('enable_offload')),
              'save_video='+str((e.get('video_cfg') or {}).get('save_video')))
    for comp in ('actor','rollout'):
        c=cfg[comp]
        print(' '+comp,
              'enable_offload='+str(c.get('enable_offload')),
              'micro_batch_size='+str(c.get('micro_batch_size')),
              'global_batch_size='+str(c.get('global_batch_size')))
PY
printf 'GPU\n'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'FAST_PROGRESS\n'
grep -anE 'Generating Rollout Epochs|Global Step:|Fatal|Traceback|OIDN|out of memory' "${FAST%/runtime/resolved.yaml}/runtime/driver.log" | tail -20 || true
