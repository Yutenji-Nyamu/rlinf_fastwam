set -euo pipefail
v4=/data/chenyiteng/projects/lerobot-sidney/lerobot-0b067df57d21
v6=/data/chenyiteng/projects/lerobot-sidney/lerobot-30da8e687a6d
echo '=== v4 pi05 config ==='
sed -n '1,260p' "$v4/src/lerobot/policies/pi05/configuration_pi05.py"
echo '=== v4 processor factory ==='
grep -R "def make_pi05_pre_post_processors\|class PI05" -n "$v4/src/lerobot/policies/pi05" "$v4/src/lerobot/policies/factory.py" | head -80
echo '=== processor registry names v4 ==='
grep -R "rename_observations_processor\|relative_actions_processor\|pi05_prepare_state_tokenizer_processor_step\|tokenizer_processor\|device_processor\|normalizer_processor\|unnormalizer_processor\|absolute_actions_processor" -n "$v4/src/lerobot" | head -120
echo '=== v4 vs v6 model files summary ==='
for f in configuration_pi05.py modeling_pi05.py processor_pi05.py; do
  sha256sum "$v4/src/lerobot/policies/pi05/$f" "$v6/src/lerobot/policies/pi05/$f" 2>/dev/null || true
done
