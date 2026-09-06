#!/usr/bin/env bash
set -euo pipefail

# Scope: validate locked local inputs, make independent ordinary copies, run the
# vendored official path updater, and create the official policy symlink.
# No network, package install, model download, simulator, or evaluator command.

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
VRT="$FW/third_party/RoboTwin"
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
NATIVE=/data/chenyiteng/projects/robotwin-native/RoboTwin
COMPAT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

FW_PIN=7faa71108368fbb3b6885649f112af607427a2d4
FW_VENDOR_TREE=b0c3bd309d95da41224191a5b089d94727317315
VENDOR_RT_PIN=bf44be51cf5717a5595ce59447f2cf5263d2aa95
VENDOR_README_BLOB=74469a70a230043baabf552245867ae49a155b27
UPDATER_BLOB=bf62165597df152077d276f07b6bd74290e5bc3b
POLICY_TREE=b5681b9db0eef9b72b17102bcc789171acf79973

NATIVE_PIN=30954692d06ba7e89f7a6b76064f4062c488fa81
ASSET_HF_REVISION=a967b852afa21a9cbf19a198f7e653109042e87c
EXPECTED_TASK_CONFIG_TREE=fdb995fb05a65f4ee6bdfaf633a89631af93db33
FORBIDDEN_COMPAT_TASK_CONFIG_TREE=f114afc63a3ffac0ac6f5ef9d780d25c0aee5a14

# Observed /data allocation increase from the previous ordinary copy of these
# exact three asset directories. It is a capacity estimate, not a content hash.
EXPECTED_ASSET_DISK_INCREMENT_BYTES=16659820544

ASSET_SOURCE="$NATIVE/assets"
ASSET_TARGET="$VRT/assets"
ASSET_STAGE="$VRT/.fw-sz-200-assets.partial"
TASK_TARGET="$VRT/task_config"
TASK_STAGE="$VRT/.fw-sz-200-task-config.partial"
POLICY_SOURCE="$FW/experiments/robotwin/fastwam_policy"
POLICY_LINK="$VRT/policy/fastwam_policy"
CUROBO_SRC="$VRT/envs/curobo"
CUROBO_REV=d64c4b005459db10c5dd867d8b30a87d5bda9bdb

die() {
  printf 'STOP: %s\n' "$*" >&2
  exit 1
}

tree_manifest() {
  local root=$1
  (cd "$root" && find . -printf '%y\t%P\t%s\t%l\n' | LC_ALL=C sort)
}

tree_manifest_sha256() {
  tree_manifest "$1" | sha256sum | awk '{print $1}'
}

count_regular_files() {
  find "$1" -type f -printf '.\n' | wc -l
}

printf 'timestamp_start=%s\n' "$(date --iso-8601=seconds)"
printf 'fw=%s\nvendor=%s\nnative=%s\ncompat_forbidden=%s\n' \
  "$FW" "$VRT" "$NATIVE" "$COMPAT"

printf '%s\n' '=== LOCKED SOURCE PREFLIGHT ==='
test -d "$FW/.git" || die "missing FastWAM Git worktree: $FW"
test -d "$NATIVE/.git" || die "missing native RoboTwin Git worktree: $NATIVE"
test -x "$ENV/bin/python" || die "missing FastWAM Python: $ENV/bin/python"
test "$(git -C "$FW" rev-parse HEAD)" = "$FW_PIN" || die 'FastWAM HEAD drift'
test "$(git -C "$FW" rev-parse HEAD:third_party/RoboTwin)" = "$FW_VENDOR_TREE" \
  || die 'vendored RoboTwin tracked tree drift'
test "$(git -C "$FW" rev-parse HEAD:third_party/RoboTwin/README.vendor.md)" = "$VENDOR_README_BLOB" \
  || die 'README.vendor.md blob drift'
test "$(git -C "$FW" rev-parse HEAD:third_party/RoboTwin/script/update_embodiment_config_path.py)" = "$UPDATER_BLOB" \
  || die 'embodiment updater blob drift'
grep -Fq "$VENDOR_RT_PIN" "$VRT/README.vendor.md" || die 'vendor upstream pin not declared'
test "$(git -C "$FW" rev-parse HEAD:experiments/robotwin/fastwam_policy)" = "$POLICY_TREE" \
  || die 'FastWAM policy tree drift'
test -d "$CUROBO_SRC/.git" || die 'missing source-locked CuRobo checkout'
test "$(git -C "$CUROBO_SRC" rev-parse HEAD)" = "$CUROBO_REV" || die 'CuRobo HEAD drift'
test -z "$(git -C "$CUROBO_SRC" status --porcelain)" || die 'CuRobo checkout is dirty'
pre_status="$(git -C "$FW" status --porcelain --untracked-files=normal)"
test -z "$pre_status" || die "unexpected FastWAM status before FW-SZ-200: $pre_status"
test "$(git -C "$NATIVE" rev-parse HEAD)" = "$NATIVE_PIN" || die 'native RoboTwin HEAD drift'
test -z "$(git -C "$NATIVE" status --porcelain --untracked-files=no -- assets)" \
  || die 'native tracked assets have changes'

# The current native checkout no longer has root task_config. Its complete Git
# clone must already contain the exact vendored upstream object; this script
# never fetches it from the network.
if git -C "$NATIVE" cat-file -e 'HEAD:task_config' 2>/dev/null; then
  die 'unexpected current native root task_config; re-audit its authority'
fi
git -C "$NATIVE" cat-file -e "${VENDOR_RT_PIN}^{commit}" 2>/dev/null \
  || die 'native clone lacks local bf44be51 vendor commit object; no fetch is attempted'
actual_task_tree="$(git -C "$NATIVE" rev-parse "${VENDOR_RT_PIN}:task_config")"
test "$actual_task_tree" = "$EXPECTED_TASK_CONFIG_TREE" \
  || die "vendor task_config tree mismatch: $actual_task_tree"
printf 'vendor_upstream_commit=%s\nvendor_task_config_tree=%s\n' \
  "$VENDOR_RT_PIN" "$actual_task_tree"

if test -d "$COMPAT/.git"; then
  compat_task_tree="$(git -C "$COMPAT" rev-parse HEAD:task_config)"
  test "$compat_task_tree" = "$FORBIDDEN_COMPAT_TASK_CONFIG_TREE" \
    || die "compatibility task_config identity changed: $compat_task_tree"
  printf 'forbidden_compat_task_config_tree=%s\n' "$compat_task_tree"
else
  printf 'forbidden_compat_tree=not-present-and-not-used\n'
fi

test "$(realpath -e "$VRT")" = "$(realpath -e "$FW")/third_party/RoboTwin" \
  || die 'vendor target escaped locked FastWAM worktree'
test -w "$VRT" || die "vendor root is not writable: $VRT"
for target in "$ASSET_TARGET" "$ASSET_STAGE" "$TASK_TARGET" "$TASK_STAGE" "$POLICY_LINK"; do
  if test -e "$target" || test -L "$target"; then
    die "refusing existing target: $target"
  fi
done
test -d "$POLICY_SOURCE" || die "missing policy source: $POLICY_SOURCE"
test -d "$VRT/policy" || die "missing vendor policy parent: $VRT/policy"

printf '%s\n' '=== ASSET SOURCE AND CAPACITY PREFLIGHT ==='
test -d "$ASSET_SOURCE" || die "missing asset source: $ASSET_SOURCE"
test ! -L "$ASSET_SOURCE" || die 'asset source root must not be a symlink'
mapfile -t asset_metadata < <(
  find "$ASSET_SOURCE/.cache/huggingface/download" -maxdepth 1 -type f -name '*.metadata' \
    | LC_ALL=C sort
)
test "${#asset_metadata[@]}" -eq 3 \
  || die "expected three official asset metadata files, got ${#asset_metadata[@]}"
for metadata in "${asset_metadata[@]}"; do
  test "$(sed -n '1p' "$metadata")" = "$ASSET_HF_REVISION" \
    || die "asset revision mismatch in $metadata"
  printf 'asset_metadata=%s revision=%s\n' "$metadata" "$ASSET_HF_REVISION"
done
declare -A expected_counts=(
  [background_texture]=11000
  [embodiments]=229
  [objects]=9368
)
source_apparent_bytes=0
source_allocated_bytes=0
source_file_count=0
for dirname in background_texture embodiments objects; do
  src="$ASSET_SOURCE/$dirname"
  test -d "$src" || die "missing official asset directory: $src"
  test ! -L "$src" || die "official asset directory is a symlink: $src"
  count="$(count_regular_files "$src")"
  test "$count" -eq "${expected_counts[$dirname]}" \
    || die "$dirname file-count mismatch: $count"
  apparent="$(du -sb "$src" | awk '{print $1}')"
  allocated="$(du -s -B1 "$src" | awk '{print $1}')"
  manifest="$(tree_manifest_sha256 "$src")"
  printf 'asset_source name=%s files=%s apparent_bytes=%s allocated_bytes=%s manifest_sha256=%s\n' \
    "$dirname" "$count" "$apparent" "$allocated" "$manifest"
  source_apparent_bytes=$((source_apparent_bytes + apparent))
  source_allocated_bytes=$((source_allocated_bytes + allocated))
  source_file_count=$((source_file_count + count))
done
read -r data_used_before data_available_before \
  < <(df -B1 --output=used,avail /data | tail -n 1)
test "$data_available_before" -ge "$EXPECTED_ASSET_DISK_INCREMENT_BYTES" \
  || die "insufficient /data capacity: $data_available_before bytes available"
printf 'asset_source_total files=%s apparent_bytes=%s allocated_bytes=%s\n' \
  "$source_file_count" "$source_apparent_bytes" "$source_allocated_bytes"
printf 'data_before used_bytes=%s available_bytes=%s expected_increment_bytes=%s\n' \
  "$data_used_before" "$data_available_before" "$EXPECTED_ASSET_DISK_INCREMENT_BYTES"

printf '%s\n' '=== ORDINARY INDEPENDENT ASSET COPY ==='
mkdir "$ASSET_STAGE"
for dirname in background_texture embodiments objects; do
  cp -a --reflink=never "$ASSET_SOURCE/$dirname" "$ASSET_STAGE/"
  diff \
    <(tree_manifest "$ASSET_SOURCE/$dirname") \
    <(tree_manifest "$ASSET_STAGE/$dirname")
done
test ! -L "$ASSET_STAGE" || die 'asset staging unexpectedly became a symlink'
asset_manifest_before_updater="$(tree_manifest_sha256 "$ASSET_STAGE")"
mv "$ASSET_STAGE" "$ASSET_TARGET"
test -d "$ASSET_TARGET" && test ! -L "$ASSET_TARGET" \
  || die 'published vendor assets are not an independent directory'
printf 'asset_manifest_before_updater=%s\n' "$asset_manifest_before_updater"

printf '%s\n' '=== VENDORED OFFICIAL EMBODIMENT PATH UPDATE ==='
mapfile -t templates < <(find "$ASSET_TARGET/embodiments" -type f -name '*_tmp.yml' | LC_ALL=C sort)
test "${#templates[@]}" -eq 6 || die "expected 6 embodiment templates, got ${#templates[@]}"
printf 'embodiment_templates=%s\n' "${#templates[@]}"
printf '%s\n' "${templates[@]}"
(cd "$VRT" && "$ENV/bin/python" script/update_embodiment_config_path.py </dev/null)
for template in "${templates[@]}"; do
  generated=${template%_tmp.yml}.yml
  test -s "$generated" || die "missing generated embodiment config: $generated"
  grep -Fq "$ASSET_TARGET" "$generated" \
    || die "generated config does not point at vendor assets: $generated"
  if grep -Fq '${ASSETS_PATH}' "$generated" || grep -Fq '$ASSETS_PATH' "$generated"; then
    die "unresolved ASSETS_PATH placeholder: $generated"
  fi
done
if grep -RIFq "$NATIVE/assets" "$ASSET_TARGET/embodiments" --include='*.yml'; then
  die 'native asset root leaked into vendor embodiment YAML'
fi
if test -d "$COMPAT/assets" \
  && grep -RIFq "$COMPAT/assets" "$ASSET_TARGET/embodiments" --include='*.yml'; then
  die 'RLinf compatibility asset root leaked into vendor embodiment YAML'
fi
asset_manifest_after_updater="$(tree_manifest_sha256 "$ASSET_TARGET")"
printf 'asset_manifest_after_updater=%s\n' "$asset_manifest_after_updater"

printf '%s\n' '=== EXACT VENDOR TASK_CONFIG MATERIALIZATION ==='
mkdir "$TASK_STAGE"
git -C "$NATIVE" archive --format=tar "${VENDOR_RT_PIN}:task_config" \
  | tar -xf - -C "$TASK_STAGE"
test "$(find "$TASK_STAGE" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 7 \
  || die 'task_config does not contain exactly seven root files'
test "$(find "$TASK_STAGE" -mindepth 1 ! -type f | wc -l)" -eq 0 \
  || die 'task_config contains an unexpected non-regular entry'
while read -r expected_blob relative_path; do
  test -f "$TASK_STAGE/$relative_path" || die "missing task_config/$relative_path"
  test ! -x "$TASK_STAGE/$relative_path" || die "unexpected executable task_config/$relative_path"
  actual_blob="$(git hash-object --no-filters "$TASK_STAGE/$relative_path")"
  test "$actual_blob" = "$expected_blob" \
    || die "blob mismatch for task_config/$relative_path: $actual_blob"
done <<'TASK_CONFIG_BLOBS'
fc83ec6a4c3ef5b3953b34a87857255843aa8d51 _camera_config.yml
de2ddc5a7ad7de477e9c51855e36e2f5d5a0510f _config_template.yml
672c88e1c1e17cc51ac806261d77bfc793bf5baa _embodiment_config.yml
afd0153aa87931f387cac391a5f71cde69c76a07 _eval_step_limit.yml
b345b27b42e1d48a212108ff5acb80d637bead10 create_task_config.sh
7e0de0aec1ad1b151570486120c7d25050be01c4 demo_clean.yml
d3fc5ed19c4d432d1e2d5696e5151f8e7b0b7adc demo_randomized.yml
TASK_CONFIG_BLOBS
mv "$TASK_STAGE" "$TASK_TARGET"
printf 'task_config_tree=%s files=7 payload_bytes=4070\n' "$EXPECTED_TASK_CONFIG_TREE"

printf '%s\n' '=== OFFICIAL POLICY LINK ==='
# Target absence was asserted above, so plain ln -s preserves fail-fast semantics
# while producing the same link requested by the official integration plan.
ln -s "$POLICY_SOURCE" "$POLICY_LINK"
test -L "$POLICY_LINK" || die 'policy link was not created'
test "$(readlink "$POLICY_LINK")" = "$POLICY_SOURCE" || die 'policy link text mismatch'
test "$(readlink -f "$POLICY_LINK")" = "$(realpath -e "$POLICY_SOURCE")" \
  || die 'policy link resolves outside current FastWAM source'

printf '%s\n' '=== POSTFLIGHT ==='
git -C "$FW" diff --quiet || die 'tracked FastWAM worktree changed'
git -C "$FW" diff --cached --quiet || die 'FastWAM index changed'
git -C "$FW" check-ignore -q third_party/RoboTwin/assets/background_texture \
  || die 'vendor assets are unexpectedly not ignored'
git -C "$FW" check-ignore -q third_party/RoboTwin/task_config/demo_clean.yml \
  || die 'vendor task_config is unexpectedly not ignored'
post_status="$(git -C "$FW" status --porcelain --untracked-files=normal)"
expected_post_status='?? third_party/RoboTwin/policy/fastwam_policy'
test "$post_status" = "$expected_post_status" \
  || die "unexpected FastWAM status after FW-SZ-200: $post_status"

read -r data_used_after data_available_after \
  < <(df -B1 --output=used,avail /data | tail -n 1)
printf 'data_after used_bytes=%s available_bytes=%s actual_used_delta_bytes=%s\n' \
  "$data_used_after" "$data_available_after" "$((data_used_after - data_used_before))"
printf 'policy_link=%s -> %s\n' "$POLICY_LINK" "$(readlink "$POLICY_LINK")"
printf 'tracked_diff=none\nexpected_untracked=%s\n' "$post_status"
printf 'timestamp_end=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' 'FW_SZ_200_VENDOR_INPUTS_OK'
