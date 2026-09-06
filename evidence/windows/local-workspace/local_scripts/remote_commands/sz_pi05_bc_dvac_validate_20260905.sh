set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc-dvac
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root:/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment"
export ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export PI0_MODEL_PATH=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
export PI05_MODEL_PATH=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1
export RAY_ADDRESS=172.17.0.1:6389 RLINF_CODE_WORKING_DIR="$root"
cd "$root"
"$venv/bin/python" -m ruff check --select I --fix tests/unit_tests/test_online_bc_dvac.py tests/unit_tests/test_pi05_online_bc_dvac.py
"$venv/bin/python" -m ruff format tests/unit_tests/test_online_bc_dvac.py tests/unit_tests/test_pi05_online_bc_dvac.py
git diff --check
CUDA_VISIBLE_DEVICES='' "$venv/bin/python" -m pytest -q tests/unit_tests/test_online_bc.py tests/unit_tests/test_online_bc_dvac.py tests/unit_tests/test_pi05_online_bc.py tests/unit_tests/test_pi05_online_bc_dvac.py
unset CUDA_VISIBLE_DEVICES
"$venv/bin/python" /data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-pi05-dvac-20260905/validate.py
git diff --stat
