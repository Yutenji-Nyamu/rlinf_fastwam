#!/usr/bin/env bash
set -u
echo '== gpu =='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '== chenyiteng relevant processes =='
ps -u chenyiteng -o pid=,ppid=,etime=,cmd= | grep -E 'ray job submit|main_embodied|wrapper\.sh|resource_observer|fastwam|sidney' | grep -v grep || true
echo '== owned gpu processes =='
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits || true
