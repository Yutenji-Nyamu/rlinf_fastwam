set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
sed -n '940,1055p' "$src/src/lerobot/policies/pi05/modeling_pi05.py"
grep -R -n -E 'from_pretrained\(.*policy_(pre|post)processor|PolicyProcessorPipeline.from_pretrained|make_pre_post_processors' "$src/src/lerobot" | head -60 || true
