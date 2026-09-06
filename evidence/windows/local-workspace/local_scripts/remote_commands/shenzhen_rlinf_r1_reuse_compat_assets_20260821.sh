#!/usr/bin/env bash
set -euo pipefail

SRC=/data/chenyiteng/projects/robotwin-native/RoboTwin
DST=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
STAGE=/data/chenyiteng/projects/rlinf-shenzhen/.staging/robotwin-assets-a967b852-20260821
RUN=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821
PYTHON=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

test "$(git -C "$SRC" rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
test "$(git -C "$DST" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test -d "$RUN"
test -x "$PYTHON"
test ! -e "$STAGE"
for dirname in background_texture embodiments objects; do
  test -d "$SRC/assets/$dirname"
  test ! -e "$DST/assets/$dirname"
done

exec > >(tee "$RUN/assets_reuse.log") 2>&1

printf '%s\n' '=== PRE-COPY ==='
standalone_status_before=$(git -C "$SRC" status --porcelain=v1 -z | sha256sum | cut -d' ' -f1)
printf 'standalone_status_sha256=%s\n' "$standalone_status_before"
printf 'compat_status='; git -C "$DST" status --short --branch
df -B1 /data
du -sh "$SRC/assets/background_texture" "$SRC/assets/embodiments" "$SRC/assets/objects"

mkdir -p "$STAGE"
for dirname in background_texture embodiments objects; do
  cp -a --reflink=always "$SRC/assets/$dirname" "$STAGE/"
done

for dirname in background_texture embodiments objects; do
  diff \
    <(cd "$SRC/assets/$dirname" && find . -printf '%y %P %s %l\n' | sort) \
    <(cd "$STAGE/$dirname" && find . -printf '%y %P %s %l\n' | sort)
done

for dirname in background_texture embodiments objects; do
  mv "$STAGE/$dirname" "$DST/assets/"
done
rmdir "$STAGE"

mapfile -t templates < <(find "$DST/assets/embodiments" -type f -name '*_tmp.yml' | sort)
test "${#templates[@]}" -gt 0
printf 'embodiment_templates=%s\n' "${#templates[@]}"
printf '%s\n' "${templates[@]}"

(cd "$DST" && "$PYTHON" script/update_embodiment_config_path.py </dev/null)

for template in "${templates[@]}"; do
  generated=${template%_tmp.yml}.yml
  test -s "$generated"
  grep -Fq "$DST/assets" "$generated"
done
if grep -RIlF "$SRC/assets" "$DST/assets/embodiments" --include='*.yml' | grep -q .; then
  printf '%s\n' 'standalone asset root leaked into compatibility yaml' >&2
  exit 1
fi

printf '%s\n' '=== POST-COPY ==='
standalone_status_after=$(git -C "$SRC" status --porcelain=v1 -z | sha256sum | cut -d' ' -f1)
printf 'standalone_status_sha256=%s\n' "$standalone_status_after"
test "$standalone_status_after" = "$standalone_status_before"
printf 'compat_status='; git -C "$DST" status --short --branch
du -sh "$DST/assets/background_texture" "$DST/assets/embodiments" "$DST/assets/objects"
df -B1 /data
printf '%s\n' 'R1_COMPAT_ASSETS_REUSE_OK'
