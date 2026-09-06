set -euo pipefail
base=/root/autodl-tmp/experiment_exports
out=$base/qam_pi0_robotwin_formal_stop247_20260801
run_root=/root/autodl-tmp/experiments
worktree=/root/autodl-tmp/RLinf_qam_pi0_robotwin
if test -e "$out"; then
  echo "OUTPUT_EXISTS=$out"
  exit 42
fi
mkdir -p "$out/runtime" "$out/launchers" "$out/checkpoint_completion"
for name in \
  qam_formal_20260731_v1 \
  qam_formal_20260801_v2 \
  qam_formal_resume25_to100_20260801_v3 \
  qam_formal_resume100_to380_20260801_v4
do
  cp -a "$base/$name/runtime" "$out/runtime/$name"
done
cp "$worktree/docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_launch_20260731_v1.sh" "$out/launchers/"
cp "$worktree/docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_launch_20260801_v2.sh" "$out/launchers/"
cp "$worktree/docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resume25_to100_launch_20260801_v3.sh" "$out/launchers/"
cp "$worktree/docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resume100_to380_launch_20260801_v4.sh" "$out/launchers/"
cp "$worktree/docs/rlinf-robotwin-pi0-qam/evidence/qam_source_resolved_20260731_formal_v1.yaml" "$out/launchers/"
cp "$worktree/docs/rlinf-robotwin-pi0-qam/evidence/qam_source_to_formal_20260731_v1.diff" "$out/launchers/"
find "$run_root" -path '*qam_formal*/*/checkpoints/global_step_*/*' -type f -printf '%p\t%s\n' | sort > "$out/checkpoint_inventory.tsv"
while IFS= read -r completion; do
  experiment=$(printf '%s' "$completion" | sed -n 's#^.*/experiments/\([^/]*\)/.*#\1#p')
  step=$(printf '%s' "$completion" | sed -n 's#^.*/checkpoints/\(global_step_[0-9]*\)/.*#\1#p')
  cp "$completion" "$out/checkpoint_completion/${experiment}_${step}.json"
done < <(find "$run_root" -path '*qam_formal*/*/checkpoints/global_step_*/actor/qam_components/complete.json' -type f | sort)
{
  date '+snapshot_time=%F %T %Z'
  echo 'stop_request=SIGTERM sent by user-authorized closeout at 2026-08-01 17:28:58 CST'
  echo 'last_complete_cycle=247'
  echo 'latest_recoverable_checkpoint=global_step_225'
  echo 'driver_pid=380841'
  echo 'driver_alive=no'
  echo 'monitor_pid=380842'
  echo 'monitor_alive=no'
  echo 'driver_exit_code=134 (explicit SIGTERM path)'
  echo 'monitor_exit_code=0'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  git -C "$worktree" rev-parse HEAD
  git -C "$worktree" status --short
} > "$out/post_stop_snapshot.txt"
printf '%s\n' \
  'This package contains lightweight runtime/config/log/resource evidence for the QAM formal chain.' \
  'Checkpoint tensors, DCP shards, replay tensors, videos, datasets, model weights, and environments remain server-side.' \
  'This package is sufficient for audit and plotting, not for standalone resume.' \
  > "$out/ARTIFACT_SCOPE.txt"
find "$out" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/file_sha256.txt"
archive=$base/qam_pi0_robotwin_formal_stop247_runtime_20260801.tar.gz
tar -C "$out" -czf "$archive" .
sha256sum "$archive" > "$archive.sha256"
printf 'OUTPUT=%s\n' "$out"
du -sh "$out" "$archive"
cat "$archive.sha256"
find "$out" -maxdepth 2 -type f -printf '%P %s\n' | sort
