set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
PACKET=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
RAY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ray
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

date '+TIME %Y-%m-%d %H:%M:%S %Z'

printf 'OWNED_PROCESSES\n'
ps -eo user,pid,ppid,pgid,etimes,rss,stat,args --sort=pid | grep -E 'fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1|62010000|3589666|3589667|3590591|3590594' | grep -v grep || true

printf 'GPU6_7\n'
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,pstate --format=csv,noheader,nounits | sed -n '7,8p'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true

printf 'RUNTIME_MARKERS\n'
find "$RUN/runtime" -maxdepth 1 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %f\n' 2>/dev/null | sort
for f in wrapper.pid owned.pgid resource_observer.pid ray_job_id.txt exit_code exit_code.txt timeout_exit_code finished_at.txt; do
  test -f "$RUN/runtime/$f" && printf '%s=' "$f" && cat "$RUN/runtime/$f"
done

printf 'RAY_NAMESPACE_ALIVE_COUNTS\n'
RAY_ADDRESS=172.17.0.1:6389 "$RAY" list actors --filter 'state=ALIVE' --format=json 2>/dev/null \
  | "$PY" -B -c 'import json,sys,collections; d=json.load(sys.stdin); print(dict(collections.Counter(x.get("ray_namespace") for x in d))); print("RLinf_1",[(x.get("name"),x.get("pid"),x.get("job_id"),x.get("state")) for x in d if x.get("ray_namespace")=="RLinf_1"])' || true

printf 'CHECKPOINT_DIRS\n'
find "$RUN" -type d -name 'global_step_*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort

printf 'GLOBAL_STEP10_TREE\n'
STEP10=$(find "$RUN" -type d -name global_step_10 -print -quit 2>/dev/null)
printf 'STEP10=%s\n' "$STEP10"
if test -n "$STEP10"; then
  find "$STEP10" -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %P\n' | sort -k4
  printf 'STEP10_TOTAL\n'
  du -sh "$STEP10"
  printf 'DCP_SIGNATURE\n'
  find "$STEP10" -type f \( -name '.metadata' -o -name '*.distcp' -o -name 'complete.json' -o -name 'manifest.json' -o -name '*state*.json' \) -printf '%s %p\n' | sort
fi

printf 'POST10_PARTIAL_CHECKPOINTS\n'
find "$RUN" -type d \( -name 'global_step_1[1-9]' -o -name 'global_step_[2-9][0-9]*' \) -print 2>/dev/null | sort
find "$RUN" -type f -newermt '2026-09-02 06:45:00 UTC' -path '*checkpoint*' -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -80

printf 'PACKET_FILES\n'
find "$PACKET" -maxdepth 2 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort

printf 'LAUNCHER_CANDIDATES\n'
find "$PACKET" "$RUN/runtime" -maxdepth 2 -type f \( -iname '*launch*' -o -iname '*resume*' -o -name 'command.txt' \) -print 2>/dev/null | while read -r f; do
  printf '\nFILE=%s\n' "$f"
  sed -n '1,260p' "$f"
done

printf 'RESOLVED_HASHES\n'
for f in "$PACKET/resolved.yaml" "$RUN/runtime/resolved.yaml"; do
  test -f "$f" && sha256sum "$f"
done

printf 'RESOLVED_RELEVANT_LEAVES\n'
RESOLVED="$RUN/runtime/resolved.yaml"
test -f "$RESOLVED" || RESOLVED="$PACKET/resolved.yaml"
"$PY" -B - "$RESOLVED" <<'PY'
import sys, yaml
p=sys.argv[1]
cfg=yaml.safe_load(open(p))
want=("resume", "max_step", "save_interval", "val_check", "log_path", "experiment_name", "checkpoint", "global_batch", "micro_batch", "update_epoch", "rollout_epoch", "group_size", "num_env", "episode_length", "chunk", "action_horizon", "inference_step", "offload", "placement", "gpu")
def walk(x, path=()):
    if isinstance(x,dict):
        for k,v in x.items(): walk(v,path+(str(k),))
    elif isinstance(x,list):
        for i,v in enumerate(x): walk(v,path+(str(i),))
    else:
        s=".".join(path).lower()
        if any(q in s for q in want): print(".".join(path),"=",repr(x))
walk(cfg)
PY

printf 'DRIVER_FINAL_BOUNDARY\n'
grep -aEn 'Global Step:|Saving checkpoint|Checkpoint saved|OIDN Error|Fatal Python error|PyGILState_Release|Traceback|RayActorError|WorkerCrashedError|NCCL|TCPStore|Finished fixed|eval.*success|fixed.*success' "$RUN/runtime/driver.log" 2>/dev/null | tail -180 || true

printf 'DRIVER_TAIL\n'
tail -n 100 "$RUN/runtime/driver.log" 2>/dev/null || true
