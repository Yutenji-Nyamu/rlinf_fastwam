set -euo pipefail

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
cd "$REPO"

printf 'HEAD_AND_TREE\n'
git rev-parse HEAD
git status --short
git ls-files -s third_party/RoboTwin | sed -n '1,20p'
printf 'SETUP_COMMON\n'
sed -n '1,260p' scripts/setup/_common.sh
printf 'INSTALL_ROBOTWIN\n'
sed -n '1,260p' scripts/setup/install_robotwin.sh
printf 'EVAL_WRAPPER\n'
sed -n '1,260p' scripts/eval_fasterwam_robotwin.sh
printf 'EVAL_FILES\n'
find experiments/robotwin -maxdepth 2 -type f -printf '%p\n' | sort
printf 'UV_CANDIDATES\n'
find /home/chenyiteng -maxdepth 5 -type f -name uv -perm -u+x -printf '%p\n' 2>/dev/null | sed -n '1,30p'
printf 'PYPROJECT_REQUIRES\n'
sed -n '1,240p' pyproject.toml
printf 'ROBOTWIN_ENV\n'
find environments/robotwin -maxdepth 2 -type f -printf '%p\n' | sort
