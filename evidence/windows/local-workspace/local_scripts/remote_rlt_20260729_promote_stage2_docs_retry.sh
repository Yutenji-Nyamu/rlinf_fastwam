#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/tmp/rlt_stage2_docs_upload_20260729_v2
manifest="${stage}/UPLOAD_SHA256SUMS"
packet=docs/rlinf-robotwin-pi0-rltoken/04_STAGE2_PRE_SMOKE_PACKET_20260729.md
ledger=docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
packet_sha=779705818a90db24308f0436274ac9bfa3e0ded0ed0bd8b57f12d85128bcffd6
ledger_sha=4265a895578887430a8b3c6550870b0b2f3b78687306b6c8901ac519d7189187

for relative in "${packet}" "${ledger}"; do
  old_sha="$(
    awk -v relative="${relative}" '$2 == relative {print $1}' "${manifest}"
  )"
  test -n "${old_sha}"
  test "$(sha256sum "${stage}/${relative}" | cut -d' ' -f1)" = "${old_sha}"
  test "$(sha256sum "${repo}/${relative}" | cut -d' ' -f1)" = "${old_sha}"
done

test "$(sha256sum "${stage}/${packet}.retry.part" | cut -d' ' -f1)" = \
  "${packet_sha}"
test "$(sha256sum "${stage}/${ledger}.retry.part" | cut -d' ' -f1)" = \
  "${ledger_sha}"
mv -- "${stage}/${packet}.retry.part" "${stage}/${packet}"
mv -- "${stage}/${ledger}.retry.part" "${stage}/${ledger}"
cp -- "${stage}/${packet}" "${repo}/${packet}"
cp -- "${stage}/${ledger}" "${repo}/${ledger}"

awk \
  -v packet="${packet}" \
  -v packet_sha="${packet_sha}" \
  -v ledger="${ledger}" \
  -v ledger_sha="${ledger_sha}" '
  $2 == packet {
    print packet_sha "  " packet
    next
  }
  $2 == ledger {
    print ledger_sha "  " ledger
    next
  }
  {print}
' "${manifest}" >"${manifest}.new"
mv -- "${manifest}.new" "${manifest}"

(
  cd "${stage}"
  sha256sum -c UPLOAD_SHA256SUMS >/dev/null
)
test "$(sha256sum "${repo}/${packet}" | cut -d' ' -f1)" = "${packet_sha}"
test "$(sha256sum "${repo}/${ledger}" | cut -d' ' -f1)" = "${ledger_sha}"
printf 'updated_manifest_sha256\t%s\n' "$(
  sha256sum "${manifest}" | cut -d' ' -f1
)"
printf '%s\n' STAGE2_DOCS_RETRY_PROMOTED
