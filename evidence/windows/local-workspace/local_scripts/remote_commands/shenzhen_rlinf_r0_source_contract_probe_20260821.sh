#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test -z "$(git -C "$ROOT" status --porcelain)"
cd "$ROOT"

printf '%s\n' '=== REQUIREMENTS FILES ==='
find requirements -maxdepth 4 -type f -printf '%p\n' | sort

printf '%s\n' '=== TARGET HASHES ==='
for file in \
  requirements/install.sh \
  docs/source-en/rst_source/examples/embodied/robotwin.rst \
  docs/source-en/rst_source/evaluations/guides/robotwin.rst \
  examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml; do
  test -f "$file"
  sha256sum "$file"
done

printf '%s\n' '=== INSTALLER RELEVANT LINES ==='
grep -nE 'venv|no-root|embodied|openpi|robotwin|uv|python|mirror|root' requirements/install.sh

printf '%s\n' '=== REQUIREMENTS REFERENCES ==='
grep -R -nE 'rlinf-openpi|openpi|robotwin|RoboTwin|uv venv|python' requirements \
  --include='*.sh' --include='*.txt' --include='*.toml' --include='*.yaml' --include='*.yml' | head -n 500

printf '%s\n' '=== OFFICIAL ROBOTWIN GUIDE COMMANDS ==='
sed -n '130,275p' docs/source-en/rst_source/examples/embodied/robotwin.rst

printf '%s\n' '=== OFFICIAL EVAL GUIDE COMMANDS ==='
sed -n '1,125p' docs/source-en/rst_source/evaluations/guides/robotwin.rst
sed -n '250,290p' docs/source-en/rst_source/evaluations/guides/robotwin.rst

printf '%s\n' '=== OFFICIAL EVAL CONFIG ==='
sed -n '1,260p' evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml

printf '%s\n' '=== OFFICIAL PPO CONFIG ==='
sed -n '1,260p' examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml
