set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python

cd "$RLT_ROOT"
export PYTHONPATH="$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" -B -c \
  "import pathlib, rlinf; print(pathlib.Path(rlinf.__file__).resolve())"

"$PYTHON_BIN" -B -m pytest -q \
  tests/unit_tests/test_robotwin_rlt_contract.py \
  tests/unit_tests/test_dsrl_transition_replay.py \
  tests/unit_tests/test_dsrl_target_shadow_resume.py
