set -u
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "$repo"
export PYTHONPATH="$repo"
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
echo '=== before ==='
date '+%F %T %Z'
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
status=0
timeout --signal=TERM --kill-after=20s 360s \
  "$venv/bin/python" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  /root/autodl-tmp/qam_two_gpu_fsdp_probe.py || status=$?
echo '=== after ==='
date '+%F %T %Z'
pgrep -af 'qam_two_gpu_fsdp_probe|torch.distributed.run' || true
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
exit "$status"
