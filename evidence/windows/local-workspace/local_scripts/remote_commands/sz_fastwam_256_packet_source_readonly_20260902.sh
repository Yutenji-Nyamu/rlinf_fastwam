set -eu
OLD_RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
OLD_PACKET=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
printf 'COMMAND\n'
cat "$OLD_PACKET/command.txt"
printf '\nLAUNCH_MANIFEST\n'
cat "$OLD_RUN/runtime/launch_manifest.txt"
printf '\nSHA_FILES\n'
cat "$OLD_PACKET/command.sha256" "$OLD_PACKET/resolved.sha256"
