set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "$repo"
export PYTHONPATH="$repo"
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
"$venv/bin/python" -u /root/autodl-tmp/qam_config_probe.py
