set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "$repo"
export PYTHONPATH="$repo:/root/autodl-tmp"
export CUDA_VISIBLE_DEVICES=
"$venv/bin/python" -u /root/autodl-tmp/qam_inspect_shared_params.py
