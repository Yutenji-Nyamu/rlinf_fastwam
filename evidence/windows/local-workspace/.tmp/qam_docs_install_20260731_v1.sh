#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
archive=/root/autodl-tmp/qam_formal_docs_sync_20260731_v1.tar
stage=/root/autodl-tmp/qam_formal_docs_sync_20260731_v1
expected_head=4a15699e10971e306ed756dcbbf8aa65632553d5

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$repo" rev-parse '@{upstream}')" = "$expected_head"
test -z "$(git -C "$repo" status --porcelain=v1)"
printf '%s  %s\n' \
  e2531aa63ae12a01b6dc334c9d196b04b7c2d434d2eafa2dc826a9a8eee55ca1 \
  "$archive" | sha256sum -c -
test ! -e "$stage"
mkdir -p "$stage"
tar -xf "$archive" -C "$stage"

(
  cd "$stage"
  sha256sum -c - <<'EOF'
e93c3df3fbc61bae74081aee0fc44eba96a6a480c3a11a1101a51f988cee5856  HANDOFF.md
4300522b789be640f522206b5a1d49c4fe7008ca840e3a2f335eb0f4ecbc69cf  docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md
7ffa4e17aaf5bfba08c84c6d0d16d3e39a50c8ba839f34c9b97ec00de3662da3  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
a3419a06f27b213a37f3c2146ff43e2616770023402587c86a21933b63135d89  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_code_commit_push_20260731.sh
c6c772ea0624a6152896a5704f02ae84f24d00cf2a21d6f0eb2a3206062c9901  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_launch_20260731_v1.sh
389c7118ac85fc5d53cf268e59e88c20b66385e885d733e595b7340d6dc95d65  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_patch_apply_20260731.sh
c26133cd7462d7c30d5779b9a6bba224209ec0781ea003fb99cc1d74e7644915  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resolved_20260731_v1.yaml
a01571f73c4c03beaabbaf71c1e33dd7ac7375e1dcad15d286cc096381c5adfb  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_schedule_server_tests_20260731.sh
45bea3edcd28d9b7d8475ce66fe7ef1cf9533dcbc795a78aa94fd210ad4310b4  docs/rlinf-robotwin-pi0-qam/evidence/qam_source_resolved_20260731_formal_v1.yaml
851dd01876ce4cfbc4893981a360eba9c11fd02bfed504791da91fcf3fb0a07c  docs/rlinf-robotwin-pi0-qam/evidence/qam_source_to_formal_20260731_v1.diff
EOF
)

files=(
  HANDOFF.md
  docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_code_commit_push_20260731.sh
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_launch_20260731_v1.sh
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_patch_apply_20260731.sh
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resolved_20260731_v1.yaml
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_schedule_server_tests_20260731.sh
  docs/rlinf-robotwin-pi0-qam/evidence/qam_source_resolved_20260731_formal_v1.yaml
  docs/rlinf-robotwin-pi0-qam/evidence/qam_source_to_formal_20260731_v1.diff
)

for file in "${files[@]}"; do
  install -D -m 0644 "$stage/$file" "$repo/$file"
done

git -C "$repo" diff --check -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_code_commit_push_20260731.sh \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_launch_20260731_v1.sh \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_patch_apply_20260731.sh \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resolved_20260731_v1.yaml \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_schedule_server_tests_20260731.sh \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_source_resolved_20260731_formal_v1.yaml

if grep -R -n -F 'NBo7SQqoatnZ' "${files[@]/#/$repo/}"; then
  printf 'SECRET_SCAN_FAIL=password\n' >&2
  exit 1
fi
if grep -R -n -E -- '-----BEGIN (OPENSSH|RSA|EC|PRIVATE) .*KEY-----' \
  "${files[@]/#/$repo/}"
then
  printf 'SECRET_SCAN_FAIL=private_key\n' >&2
  exit 1
fi

git -C "$repo" status --short
