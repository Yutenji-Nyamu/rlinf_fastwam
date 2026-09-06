set -euo pipefail
v4=/data/chenyiteng/projects/lerobot-sidney/lerobot-0b067df57d21
echo '=== PI05 class ==='
sed -n '880,1040p' "$v4/src/lerobot/policies/pi05/modeling_pi05.py"
echo '=== generic from_pretrained/load ==='
grep -R "def from_pretrained" -n "$v4/src/lerobot/policies/pretrained.py" "$v4/src/lerobot" | head -30
sed -n '80,240p' "$v4/src/lerobot/policies/pretrained.py" 2>/dev/null || true
echo '=== make processors ==='
sed -n '90,200p' "$v4/src/lerobot/policies/pi05/processor_pi05.py"
echo '=== action API ==='
grep -n "def select_action\|def predict_action_chunk\|action_queue" "$v4/src/lerobot/policies/pi05/modeling_pi05.py" | tail -30
