set -u
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
echo '=== command ==='
cat "$RUN/runtime/command.txt"
echo '=== wrapper ==='
cat "$RUN/runtime/wrapper.sh"
echo '=== launch manifest ==='
cat "$RUN/runtime/launch_manifest.txt"
echo '=== source head ==='
cat "$RUN/runtime/source_head.txt"
echo '=== runtime dir ==='
find "$RUN/runtime" -maxdepth 1 -type f -printf '%f %s\n' | sort
echo '=== robotwin root topology ==='
RT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
find "$RT" -maxdepth 1 -mindepth 1 -printf '%f %y %l\n' | sort | head -n 120
du -sh "$RT" 2>/dev/null || true
