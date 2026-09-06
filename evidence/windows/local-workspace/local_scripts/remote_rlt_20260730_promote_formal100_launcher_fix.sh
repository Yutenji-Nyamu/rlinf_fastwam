#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime
target="${runtime}/launch_background.sh"
candidate="${runtime}/launch_background.sh.part"
archive="${runtime}/launch_background.failed_self_match.sh"
run=/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1
old_sha=a71c36d46ab8d99b47956efc79400ad4dc8c6cb0c98ec6dc7ac4b79134f89509
new_sha=a54c70c2c4691836d50f8663cf29e524ae9f2a57c1aefe80938d4e42c86c8537

test ! -e "${run}"
for name in driver_pid.txt monitor_pid.txt started_at.txt exit_code.txt finished_at.txt driver.log resources.csv; do
  test ! -e "${runtime}/${name}"
done
test ! -e "${archive}"
test "$(sha256sum "${target}" | cut -d' ' -f1)" = "${old_sha}"
test "$(sha256sum "${candidate}" | cut -d' ' -f1)" = "${new_sha}"
bash -n "${candidate}"

OLD="${target}" NEW="${candidate}" \
  /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import os
from pathlib import Path

old = Path(os.environ["OLD"]).read_text().splitlines()
new = Path(os.environ["NEW"]).read_text().splitlines()
assert len(old) == len(new)
changes = [(i + 1, a, b) for i, (a, b) in enumerate(zip(old, new)) if a != b]
assert changes == [
    (
        14,
        "  pgrep -af 'train_embodied_agent.py|rlt_stage2_formal_100c|raylet|gcs_server' \\",
        "  pgrep -af 'train_embodied_agent.py|raylet|gcs_server' \\",
    )
]
PY

mapfile -t active_rows < <(
  pgrep -af 'train_embodied_agent.py|raylet|gcs_server' || true
)
test "${#active_rows[@]}" = 0
mapfile -t compute_rows < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk 'NF'
)
test "${#compute_rows[@]}" = 0

cp "${target}" "${archive}"
cat >"${runtime}/launch_failure_self_match.txt" <<EOF
failed_at=$(date --iso-8601=seconds)
exit_code=1
run_root_created=no
driver_started=no
reason=process gate matched launch script argv through rlt_stage2_formal_100c
old_sha256=${old_sha}
fixed_sha256=${new_sha}
EOF
chmod 600 "${runtime}/launch_failure_self_match.txt"
mv "${candidate}" "${target}"
chmod 700 "${target}"
test "$(sha256sum "${target}" | cut -d' ' -f1)" = "${new_sha}"

printf 'promoted_at\t%s\n' "$(date --iso-8601=seconds)"
printf 'old_sha256\t%s\n' "${old_sha}"
printf 'new_sha256\t%s\n' "${new_sha}"
printf 'archive\t%s\n' "${archive}"
printf '%s\n' RLT_STAGE2_FORMAL100_LAUNCHER_FIX_PROMOTED
