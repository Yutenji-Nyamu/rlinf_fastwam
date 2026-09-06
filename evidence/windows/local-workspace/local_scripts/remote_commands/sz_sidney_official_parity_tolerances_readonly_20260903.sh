set -eu
P=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat/tests/policies/pi0_pi05/utils/openpi_parity.py
grep -n -A110 -B10 'def assert_processor_inputs_match_lerobot' "$P"
grep -n -A80 -B10 'def make_openpi_observation_from_raw' "$P"
