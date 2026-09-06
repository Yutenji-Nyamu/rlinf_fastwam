#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

fresh=/root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_fresh.sh
resume=/root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_resume.sh
fresh_part="${fresh}.v3.part"
resume_part="${resume}.v3.part"

test "$(sha256sum "${fresh}" | cut -d' ' -f1)" = \
  13ea8602d257bc084b3441ecd7a2220dcfb2d0aab6869dd7fb5ac0445fdeb7c6
test "$(sha256sum "${resume}" | cut -d' ' -f1)" = \
  b76b67c47cd1e629990157ad5c4f1d8628bd253a26cf533989aadfde166b896c
test "$(sha256sum "${fresh_part}" | cut -d' ' -f1)" = \
  6e0f1c7ce5497bd3d5a2bef539bbea5e3fc964a5d8259b16f472cf353d19e27a
test "$(sha256sum "${resume_part}" | cut -d' ' -f1)" = \
  6494eaeb8cb2e6c1decee07d97798a0aba368ee15b0491faf3ecdfe6a9ff054c
bash -n "${fresh_part}" "${resume_part}"

mv -- "${fresh_part}" "${fresh}"
mv -- "${resume_part}" "${resume}"
chmod 0755 "${fresh}" "${resume}"

bash -n "${fresh}" "${resume}"
test "$(sha256sum "${fresh}" | cut -d' ' -f1)" = \
  6e0f1c7ce5497bd3d5a2bef539bbea5e3fc964a5d8259b16f472cf353d19e27a
test "$(sha256sum "${resume}" | cut -d' ' -f1)" = \
  6494eaeb8cb2e6c1decee07d97798a0aba368ee15b0491faf3ecdfe6a9ff054c
printf '%s\n' STAGE2_LAUNCHERS_V3_PROMOTED
