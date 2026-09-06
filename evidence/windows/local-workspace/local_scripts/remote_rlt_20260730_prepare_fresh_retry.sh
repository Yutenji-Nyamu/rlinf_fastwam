#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
smoke=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
evidence=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1
smoke_failed=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1_failed_launcher_127
evidence_failed=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1_failed_launcher_127
launcher=/root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_fresh.sh
launcher_part="${launcher}.fixed_20260730.part"
old_sha=6e0f1c7ce5497bd3d5a2bef539bbea5e3fc964a5d8259b16f472cf353d19e27a
new_sha=473f339af5123802526dae93fe2fde7289fe52f32efb80b581ded073eaabd985

test "$(git -C "${repo}" branch --show-current)" = codex/rlt-pi0-robotwin
test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
test "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})" = \
  $'0\t0'
test ! -e "${smoke}"
test -d "${evidence}"
test ! -L "${evidence}"
test ! -e "${smoke_failed}"
test ! -e "${evidence_failed}"
test "$(cat "${evidence}/fresh_runtime/exit_code.txt")" = 127
grep -Fq "No such file or directory" "${evidence}/fresh_runtime/driver.log"
test "$(sha256sum "${launcher}" | cut -d' ' -f1)" = "${old_sha}"
test "$(sha256sum "${launcher_part}" | cut -d' ' -f1)" = "${new_sha}"
bash -n "${launcher_part}"

case "$(realpath -m "${evidence}")" in
  /root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1) ;;
  *) exit 31 ;;
esac
case "$(realpath -m "${evidence_failed}")" in
  /root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1_failed_launcher_127) ;;
  *) exit 32 ;;
esac

cp -- "${launcher}" "${evidence}/fresh_runtime/launcher_failed_v1.sh"
sha256sum "${evidence}/fresh_runtime/launcher_failed_v1.sh" \
  >"${evidence}/fresh_runtime/launcher_failed_v1.sh.sha256"
mv -- "${evidence}" "${evidence_failed}"
mv -- "${launcher_part}" "${launcher}"
chmod 0755 "${launcher}"

test ! -e "${evidence}"
test "$(sha256sum "${launcher}" | cut -d' ' -f1)" = "${new_sha}"
bash -n "${launcher}"
printf 'failed_smoke\t%s\n' NOT_CREATED
printf 'failed_evidence\t%s\n' "${evidence_failed}"
printf 'fixed_launcher_sha256\t%s\n' "${new_sha}"
printf '%s\n' RLT_FRESH_RETRY_PREPARED
