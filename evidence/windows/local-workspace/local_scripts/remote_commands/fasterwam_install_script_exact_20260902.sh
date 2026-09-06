set -euo pipefail

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
printf 'INSTALL_ROBOTWIN_SH\n'
nl -ba "$REPO/scripts/setup/install_robotwin.sh"
printf 'PATCH_ROBOTWIN_SH\n'
nl -ba "$REPO/scripts/setup/patch_robotwin_env.py"
printf 'COMMON_SH\n'
nl -ba "$REPO/scripts/setup/_common.sh"
