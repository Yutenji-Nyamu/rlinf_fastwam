set -euo pipefail
for tag in v0.4.3 v0.5.0; do
  tmp=$(mktemp -d /data/chenyiteng/tmp/lerobot-version-XXXXXX)
  git clone -q --depth 1 --branch "$tag" https://github.com/huggingface/lerobot.git "$tmp/lerobot"
  echo "=== $tag $(git -C "$tmp/lerobot" rev-parse HEAD) ==="
  grep -n '^requires-python\|^version =\|transformers-dep\|^pi =' "$tmp/lerobot/pyproject.toml" || true
  test -f "$tmp/lerobot/src/lerobot/policies/pi05/modeling_pi05.py" && echo PI05=yes || echo PI05=no
  test -f "$tmp/lerobot/src/lerobot/envs/robotwin.py" && echo ROBOTWIN=yes || echo ROBOTWIN=no
  test -f "$tmp/lerobot/docs/source/robotwin.mdx" && echo ROBOTWIN_DOC=yes || true
  if [ -f "$tmp/lerobot/src/lerobot/policies/pi05/modeling_pi05.py" ]; then
    grep -R "make_pre_post_processors\|policy_preprocessor_step" -n "$tmp/lerobot/src/lerobot/policies" | head -20 || true
  fi
  rm -rf "$tmp"
done
echo '=== gpu index uuid ==='
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
echo '=== apps ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
