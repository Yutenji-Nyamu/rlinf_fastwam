set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
SESSION=/tmp/ray/session_2026-07-28_18-54-21_566345_70062
CKPT_ROOT=$RUN/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_formal_v1/checkpoints
EXPORT_ROOT=/root/autodl-tmp/experiment_exports/dsrl_pi0_robotwin_formal_v1_20260729
STAGE=$EXPORT_ROOT/runtime_step198
ARCHIVE=$EXPORT_ROOT/dsrl_pi0_robotwin_formal_v1_runtime_step198_20260729.tar.gz

test ! -e "$EXPORT_ROOT"
mkdir -p "$STAGE/run" "$STAGE/ray_logs" "$STAGE/metadata" "$STAGE/code_key_files"

find "$RUN" -maxdepth 1 -type f -exec cp -a -t "$STAGE/run" {} +
cp -a "$RUN/resource_monitor" "$STAGE/run/"
cp -a "$RUN/tensorboard" "$STAGE/run/"
cp -a "$SESSION/logs/." "$STAGE/ray_logs/"

{
  echo "DSRL pi0 RoboTwin formal v1 runtime bundle"
  echo "created_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "run_root=$RUN"
  echo "last_completed_cycle=198"
  echo "tensorboard_metrics_last_flushed_cycle=197"
  echo "resume_checkpoint=$CKPT_ROOT/global_step_195"
  echo "resume_checkpoint_note=step 196-198 metrics are preserved but their parameter updates are not in DCP195"
  echo "checkpoint_payloads_in_archive=no"
  echo "checkpoint_payloads_server_side=yes"
  echo "repo=$REPO"
  echo "branch=$(git -C "$REPO" branch --show-current)"
  echo "head=$(git -C "$REPO" rev-parse HEAD)"
  echo "upstream=$(git -C "$REPO" rev-parse '@{upstream}')"
} > "$STAGE/README_RUNTIME_BUNDLE.txt"

{
  echo -e "bytes\tmtime\tpath"
  find "$CKPT_ROOT" -type f \
    -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%P\n' | sort
} > "$STAGE/metadata/checkpoint_file_manifest.tsv"

{
  for checkpoint in "$CKPT_ROOT"/global_step_*
  do
    test -d "$checkpoint" || continue
    printf '%s\t%s\t%s\t%s\n' \
      "$(basename "$checkpoint")" \
      "$(du -sb "$checkpoint" | awk '{print $1}')" \
      "$(find "$checkpoint" -type f | wc -l)" \
      "$(find "$checkpoint" \( -name '*.tmp' -o -name '.metadata.tmp' \) | wc -l)"
  done
} > "$STAGE/metadata/checkpoint_summary.tsv"

find "$CKPT_ROOT" -type f \
  \( -name '.metadata' -o -name '*.metadata' -o -name 'metadata.json' \) \
  -print0 | sort -z | xargs -0 -r sha256sum \
  > "$STAGE/metadata/checkpoint_metadata_sha256.txt"

git -C "$REPO" status --short > "$STAGE/metadata/git_status.txt"
git -C "$REPO" log --reverse --date=iso-strict \
  --format='%H%x09%ad%x09%an%x09%s' \
  "$(git -C "$REPO" rev-parse 6817c73b298ff9df78d371d4b139e4e0fa8ea529^)"..HEAD \
  > "$STAGE/metadata/git_log_from_preimplementation.tsv"
git -C "$REPO" diff --binary \
  "$(git -C "$REPO" rev-parse 6817c73b298ff9df78d371d4b139e4e0fa8ea529^)"..HEAD \
  > "$STAGE/metadata/dsrl_branch_from_preimplementation.patch"

(
  cd "$REPO"
  cp --parents \
    examples/embodiment/config/robotwin_adjust_bottle_dsrl_openpi.yaml \
    examples/embodiment/config/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke.yaml \
    rlinf/models/embodiment/openpi/openpi_action_model.py \
    rlinf/workers/actor/fsdp_sac_policy_worker.py \
    rlinf/data/replay_buffer.py \
    "$STAGE/code_key_files"
)

{
  uname -a
  nvidia-smi
  /root/autodl-tmp/RLinf/.venv/bin/python --version
  /root/autodl-tmp/RLinf/.venv/bin/python -m pip freeze
} > "$STAGE/metadata/environment_snapshot.txt" 2>&1

if grep -R -I -l -E \
  'SEETA_SSH_PASSWORD|OPENAI_API_KEY|WANDB_API_KEY|HF_TOKEN|BEGIN (RSA|OPENSSH) PRIVATE KEY' \
  "$STAGE" > "$EXPORT_ROOT/secret_scan_matches.txt"
then
  echo "SECRET_SCAN=FAILED"
  cat "$EXPORT_ROOT/secret_scan_matches.txt"
  exit 3
fi
echo "SECRET_SCAN=PASS" > "$EXPORT_ROOT/secret_scan_status.txt"

tar -C "$EXPORT_ROOT" -czf "$ARCHIVE" runtime_step198
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

echo "BUNDLE_BUILD=PASS"
echo "EXPORT_ROOT=$EXPORT_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")"
cat "$ARCHIVE.sha256"
du -sh "$STAGE"
