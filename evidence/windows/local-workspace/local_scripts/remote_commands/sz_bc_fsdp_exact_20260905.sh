#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
sed -n '1,134p' "$root/rlinf/hybrid_engines/fsdp/strategy/fsdp.py"
sed -n '108,215p' "$root/rlinf/workers/actor/embodied_fsdp_actor_worker.py"
sed -n '1,260p' "$root/rlinf/hybrid_engines/fsdp/strategy/checkpoint.py"
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
grep -n -E 'def _use_(sharded|unsharded)_views|def _unshard|_deregister_orig_params|_register_orig_params' "$venv/lib/python3.11/site-packages/torch/distributed/fsdp/_flat_param.py"
sed -n '1870,2005p' "$venv/lib/python3.11/site-packages/torch/distributed/fsdp/_flat_param.py"
