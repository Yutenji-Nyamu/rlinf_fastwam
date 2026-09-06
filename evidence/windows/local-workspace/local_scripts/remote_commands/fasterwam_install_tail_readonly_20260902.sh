#!/usr/bin/env bash
set -euo pipefail
repo=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
cd "$repo"
nl -ba scripts/setup/install_robotwin.sh | sed -n '1,260p'
printf '\nVENV\n'
readlink -f .venvs/robotwin/bin/python
printf '\nCUROBO\n'
git -C third_party/RoboTwin/envs/curobo rev-parse HEAD
git -C third_party/RoboTwin/envs/curobo status --short
printf '\nPACKAGES\n'
.venvs/robotwin/bin/python -m pip show curobo 2>/dev/null || true
