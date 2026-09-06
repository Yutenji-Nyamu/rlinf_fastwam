set -euo pipefail
export PYTHONPATH=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin${PYTHONPATH:+:$PYTHONPATH}
venv=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
echo '=== GPU4 ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader)" || true
CUDA_VISIBLE_DEVICES=4 "$venv/bin/python" - <<'PY'
import lerobot.envs.robotwin as rw
import lerobot.scripts.lerobot_eval as ev
print('ROBOTWIN_IMPORT_OK',rw.__file__)
print('EVAL_IMPORT_OK',ev.__file__)
PY
