set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python
EVIDENCE=docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729
ALT_INDEX=$(mktemp /root/autodl-tmp/tmp/rlt_stage1_docs_index.XXXXXX)
EXPECTED=$(mktemp /root/autodl-tmp/tmp/rlt_stage1_docs_expected.XXXXXX)
ACTUAL=$(mktemp /root/autodl-tmp/tmp/rlt_stage1_docs_actual.XXXXXX)

cleanup() {
  rm -f -- "$ALT_INDEX" "$EXPECTED" "$ACTUAL"
}
trap cleanup EXIT

cd "$RLT_ROOT"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = e4127fd49e38362161eac08c551a7a98c11e9802

cat > "$EXPECTED" <<'EOF'
HANDOFF.md
examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md
docs/rlinf-robotwin-pi0-rltoken/02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md
docs/rlinf-robotwin-pi0-rltoken/evidence/DISK_AUDIT_COMMANDS_20260729.md
docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/README.md
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/exact_commands.txt
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/exact_commands_addendum.md
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/lr_scheduler_contract.json
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1a_driver.log
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1a_reload_driver.log
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1a_reload_resources.csv
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1a_resources.csv
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1b_driver.log
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/runtime/s1b_resources.csv
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/s1a_resolved.yaml
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/s1b_resolved.yaml
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/source_configs/robotwin_rlt_stage1_sft_openpi.yaml
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/source_configs/robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke.yaml
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/stage1_postcheck.json
EOF
sort -o "$EXPECTED" "$EXPECTED"

GIT_INDEX_FILE="$ALT_INDEX" git read-tree HEAD
while IFS= read -r path; do
  GIT_INDEX_FILE="$ALT_INDEX" git add -f -- "$path"
done < "$EXPECTED"
GIT_INDEX_FILE="$ALT_INDEX" git diff --cached --name-only | sort > "$ACTUAL"
diff -u "$EXPECTED" "$ACTUAL"
GIT_INDEX_FILE="$ALT_INDEX" git diff --cached --check

"$PYTHON_BIN" -B - "$RLT_ROOT" "$EXPECTED" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = [
    root / line
    for line in Path(sys.argv[2]).read_text().splitlines()
    if line.strip()
]
for path in paths:
    if not path.is_file():
        raise RuntimeError(f"missing file: {path}")
    if path.stat().st_size > 1024 * 1024:
        raise RuntimeError(f"evidence file exceeds 1 MiB: {path}")
    path.read_text(encoding="utf-8", errors="strict")

markdown = [path for path in paths if path.suffix == ".md"]
link_pattern = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)")
for path in markdown:
    text = path.read_text(encoding="utf-8")
    for match in link_pattern.finditer(text):
        target = match.group(1).split("#", 1)[0]
        if target and not (path.parent / target).resolve().exists():
            raise RuntimeError(f"{path}: missing relative link {target}")

credential = re.compile(
    r"(?i)(password\s*=|api[_-]?key\s*=|authorization:\s*bearer|"
    r"SEETA_SSH_PASSWORD\s*=\s*[^<])"
)
for path in paths:
    if credential.search(path.read_text(encoding="utf-8")):
        raise RuntimeError(f"credential-like content in {path}")

post = json.loads(
    (
        root
        / "docs/rlinf-robotwin-pi0-rltoken/evidence/"
        "stage1_smoke_20260729/stage1_postcheck.json"
    ).read_text()
)
assert post["resource"]["s1b"]["gpu0_peak_mib"] == 26447.0
assert post["resource"]["s1b"]["gpu1_peak_mib"] == 26447.0
assert post["checkpoint_total_gib"] > 20
assert post["s1b_checkpoint_file_count"] == 0
assert len(post["metrics"]["s1a"]) == 2
assert len(post["metrics"]["s1b"]) == 1
lr_contract = json.loads(
    (
        root
        / "docs/rlinf-robotwin-pi0-rltoken/evidence/"
        "stage1_smoke_20260729/lr_scheduler_contract.json"
    ).read_text()
)
assert lr_contract["resolved_optim"]["min_lr_rate"] == 0.1
assert lr_contract["resolved_optim"]["min_lr_present"] is False
assert (
    lr_contract["fixed_formal_2k"]["selected_lr_after_scheduler_step"]["2000"]
    == 2.5e-6
)
print(f"UTF8_LINK_CREDENTIAL_JSON_OK files={len(paths)}")
PY

sha256sum \
  "$EVIDENCE/s1a_resolved.yaml" \
  "$EVIDENCE/s1b_resolved.yaml" \
  "$EVIDENCE/exact_commands.txt" \
  "$EVIDENCE/exact_commands_addendum.md" \
  "$EVIDENCE/lr_scheduler_contract.json" \
  "$EVIDENCE/stage1_postcheck.json"
GIT_INDEX_FILE="$ALT_INDEX" git diff --cached --stat
printf '%s\n' PRECOMMIT_REVIEW_OK
