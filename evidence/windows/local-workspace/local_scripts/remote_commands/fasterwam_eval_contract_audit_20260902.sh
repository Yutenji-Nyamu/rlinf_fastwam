set -euo pipefail

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
cd "$REPO"

printf 'ROBOTWIN_PYPROJECT\n'
sed -n '1,280p' environments/robotwin/pyproject.toml
printf 'MANAGER\n'
sed -n '1,360p' experiments/robotwin/run_robotwin_manager.py
printf 'SINGLE_EVAL\n'
sed -n '1,420p' experiments/robotwin/eval_robotwin_single.py
printf 'DEPLOY_POLICY\n'
sed -n '1,420p' experiments/robotwin/fasterwam_policy/deploy_policy.py
printf 'TASK_CONFIG\n'
sed -n '1,280p' configs/task/robotwin_fasterwam_3cam_384_1e-4.yaml
printf 'MODEL_CONFIG\n'
sed -n '1,300p' configs/model/fasterwam.yaml
printf 'BASE_TRAIN_CONFIG\n'
sed -n '1,260p' configs/train.yaml
