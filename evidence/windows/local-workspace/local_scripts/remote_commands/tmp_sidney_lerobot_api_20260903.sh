set -euo pipefail
LR=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
cd "$LR"
find . -maxdepth 2 -type d | head -40
grep -R -n -E "def make_pre_post_processors|postprocessor\(|predict_action_chunk|class PI05Policy" src/lerobot tests/policies/pi0_pi05 tests/processor/test_pi05_processor.py | head -120 || true
echo '=== factory ==='
sed -n '350,470p' src/lerobot/policies/factory.py
echo '=== parity test core ==='
sed -n '170,270p' tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py
echo '=== pi05 policy ==='
grep -n -E "class PI05Policy|def predict_action_chunk|def _preprocess_images" src/lerobot/policies/pi0/modeling_pi0.py
sed -n '900,1040p' src/lerobot/policies/pi0/modeling_pi0.py
