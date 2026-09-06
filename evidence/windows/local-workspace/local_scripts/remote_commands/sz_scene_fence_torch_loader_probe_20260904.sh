#!/usr/bin/env bash
set -euo pipefail
OLD=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1
source "$OLD/runtime/environment.sh"
export MODEL=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt
export CUDA_VISIBLE_DEVICES=''
date -Is
for mode in plain preload late_load; do
 printf '\nMODE=%s\n' "$mode"
 (
  if [ "$mode" = preload ]; then source /home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/enable.sh; fi
  export PROBE_MODE=$mode
  timeout 45 "$VIRTUAL_ENV/bin/python" - <<'PY'
import os, ctypes, pathlib, traceback, zipfile
import torch
if os.environ['PROBE_MODE']=='late_load':
 site=pathlib.Path(torch.__file__).parents[1]
 for n in ['libOpenImageDenoise_core.so.2.0.1','libOpenImageDenoise.so.2.0.1']:
  ctypes.CDLL(str(site/'sapien/oidn_library'/n), mode=ctypes.RTLD_LOCAL)
 ctypes.CDLL('/home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/librlinf_scene_fence.so',mode=ctypes.RTLD_GLOBAL)
try:
 r=torch._C.PyTorchFileReader(os.environ['MODEL'])
 print('TORCH_VERSION_RECORD',r.get_record('version'))
 print('TORCH_RECORD_COUNT',len(r.get_all_records()))
except Exception:traceback.print_exc()
with zipfile.ZipFile(os.environ['MODEL']) as z:
 name=[n for n in z.namelist() if n.endswith('/version')][0]
 print('PYTHON_ZIP_VERSION',z.read(name))
PY
 )
done
R=$REPO_PATH
sed -n '1,125p' "$R/rlinf/envs/robotwin/robotwin_env.py"
rg -n 'create_worker|env_vars|EnvWorker|create_worker_group|launch' "$R/examples/embodiment/train_embodied_agent.py" "$R/rlinf/workers/env/env_worker.py" "$R/rlinf/scheduler/worker/worker.py" || true
nm -D --defined-only "$VIRTUAL_ENV/lib/python3.11/site-packages/sapien.libs/libsvulkan2.so" | grep -E 'mz_zip|mz_inflate|crc32|tinfl|tdefl' || true
date -Is
