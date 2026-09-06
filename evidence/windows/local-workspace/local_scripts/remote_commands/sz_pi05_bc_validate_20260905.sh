set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root:/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment" ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export PI0_MODEL_PATH=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
export PI05_MODEL_PATH=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1
export RAY_ADDRESS=172.17.0.1:6389 RLINF_CODE_WORKING_DIR="$root"
cd "$root"
"$venv/bin/python" -m ruff check --select I --fix tests/unit_tests/test_pi05_online_bc.py
"$venv/bin/python" -m ruff format tests/unit_tests/test_pi05_online_bc.py
CUDA_VISIBLE_DEVICES='' "$venv/bin/python" -m pytest -q tests/unit_tests/test_online_bc.py tests/unit_tests/test_pi05_online_bc.py
"$venv/bin/python" -u /data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-pi05-20260905/pi05_bc_validate_20260905.py
git diff --check
git diff --stat
