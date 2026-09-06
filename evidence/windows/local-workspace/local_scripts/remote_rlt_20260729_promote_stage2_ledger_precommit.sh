#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/tmp/rlt_stage2_docs_upload_20260729_v2
manifest="${stage}/UPLOAD_SHA256SUMS"
relative=docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
part="${stage}/${relative}.precommit.part"
new_sha=9c10b5792a05744c9ec4f891e3ba837be81c42dfcc9d41a1bb166748f4c59a75

old_sha="$(
  awk -v relative="${relative}" '$2 == relative {print $1}' "${manifest}"
)"
test -n "${old_sha}"
test "$(sha256sum "${stage}/${relative}" | cut -d' ' -f1)" = "${old_sha}"
test "$(sha256sum "${repo}/${relative}" | cut -d' ' -f1)" = "${old_sha}"
test "$(sha256sum "${part}" | cut -d' ' -f1)" = "${new_sha}"

mv -- "${part}" "${stage}/${relative}"
cp -- "${stage}/${relative}" "${repo}/${relative}"
awk -v relative="${relative}" -v new_sha="${new_sha}" '
  $2 == relative {
    print new_sha "  " relative
    next
  }
  {print}
' "${manifest}" >"${manifest}.new"
mv -- "${manifest}.new" "${manifest}"

(
  cd "${stage}"
  sha256sum -c UPLOAD_SHA256SUMS >/dev/null
)
test "$(sha256sum "${repo}/${relative}" | cut -d' ' -f1)" = "${new_sha}"
printf 'updated_manifest_sha256\t%s\n' "$(
  sha256sum "${manifest}" | cut -d' ' -f1
)"
printf '%s\n' STAGE2_LEDGER_PRECOMMIT_PROMOTED
