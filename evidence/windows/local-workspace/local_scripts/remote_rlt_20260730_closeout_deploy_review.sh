#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/tmp/rlt_stage2_fresh_closeout_20260730_v3
manifest="${stage}/UPLOAD_SHA256SUMS"
expected_head=6fd3ee7106fb82f06eda82603c41a09767151709
expected_manifest_sha=d48f1bafb564f3877b31b8891984bb5faa45c568252cc2a9bbad393aa507a54c

cd "${repo}"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = "${expected_head}"
read -r left right < <(git rev-list --left-right --count '@{upstream}...HEAD')
test "${left}" = 0
test "${right}" = 0
test "$(sha256sum "${manifest}" | cut -d' ' -f1)" = "${expected_manifest_sha}"
test "$(wc -l < "${manifest}")" = 21
(
  cd "${stage}"
  sha256sum -c UPLOAD_SHA256SUMS >/dev/null
)

git ls-files --modified --deleted --others --exclude-standard \
  | LC_ALL=C sort -u > /tmp/rlt_closeout_pre_dirty_paths.txt
cut -d' ' -f3- "${manifest}" | LC_ALL=C sort -u \
  > /tmp/rlt_closeout_manifest_paths.txt
if comm -23 \
  /tmp/rlt_closeout_pre_dirty_paths.txt \
  /tmp/rlt_closeout_manifest_paths.txt \
  | grep -q .; then
  printf '%s\n' 'unexpected pre-existing dirty paths:'
  comm -23 \
    /tmp/rlt_closeout_pre_dirty_paths.txt \
    /tmp/rlt_closeout_manifest_paths.txt
  exit 1
fi
git diff --cached --name-only | LC_ALL=C sort -u \
  > /tmp/rlt_closeout_pre_staged_paths.txt
if comm -23 \
  /tmp/rlt_closeout_pre_staged_paths.txt \
  /tmp/rlt_closeout_manifest_paths.txt \
  | grep -q .; then
  printf '%s\n' 'unexpected pre-existing staged paths:'
  comm -23 \
    /tmp/rlt_closeout_pre_staged_paths.txt \
    /tmp/rlt_closeout_manifest_paths.txt
  exit 1
fi

while read -r digest relative; do
  test -n "${digest}"
  test -n "${relative}"
  case "${relative}" in
    HANDOFF.md|docs/rlinf-robotwin-pi0-rltoken/*) ;;
    *)
      printf 'refusing path outside RLT docs: %s\n' "${relative}"
      exit 1
      ;;
  esac
  install -D -m 0644 "${stage}/${relative}" "${repo}/${relative}"
done < "${manifest}"

REPO="${repo}" MANIFEST="${manifest}" \
  /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
from pathlib import Path
import json
import os
import re

repo = Path(os.environ["REPO"])
manifest = Path(os.environ["MANIFEST"])
paths = [
    line.split("  ", 1)[1]
    for line in manifest.read_text().splitlines()
]
assert len(paths) == 21
for rel in paths:
    (repo / rel).read_bytes().decode("utf-8", errors="strict")

for rel in paths:
    path = repo / rel
    if path.suffix == ".json":
        json.loads(path.read_text())
    if path.suffix in {".md", ".yaml", ".yml", ".json", ".tsv"}:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.rstrip(" \t") != line:
                raise AssertionError(f"trailing whitespace: {rel}:{number}")

markdown = [repo / rel for rel in paths if rel.endswith(".md")]
for path in markdown:
    text = path.read_text()
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        resolved = (path.parent / target).resolve()
        assert resolved.exists(), f"missing link {path}:{target}"

fresh = repo / (
    "docs/rlinf-robotwin-pi0-rltoken/evidence/"
    "stage2_fresh_smoke_20260730"
)
summary = json.loads((fresh / "postcheck_summary.json").read_text())
assert summary["result"] == "fresh_passed_resume_not_run"
assert summary["run"]["exit_code"] == 0
assert summary["contract"]["saved_update_step"] == 8
assert summary["scope"] == {
    "fresh": True,
    "resume": False,
    "formal_started": False,
}
completion = json.loads(
    (fresh / "rlt_trainer_state_complete.json").read_text()
)
assert completion["complete"] is True
assert completion["update_step"] == 8
PY

(
  cd "${repo}/docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_fresh_smoke_20260730"
  sha256sum -c SHA256SUMS.txt >/dev/null
)

if grep -RInaE \
  --include='*.md' --include='*.json' --include='*.yaml' --include='*.tsv' \
  'BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}' \
  HANDOFF.md docs/rlinf-robotwin-pi0-rltoken; then
  printf '%s\n' 'credential-like content detected'
  exit 1
fi

git ls-files --modified --deleted --others --exclude-standard \
  | LC_ALL=C sort -u > /tmp/rlt_closeout_dirty_paths.txt
if comm -23 \
  /tmp/rlt_closeout_dirty_paths.txt \
  /tmp/rlt_closeout_manifest_paths.txt \
  | grep -q .; then
  printf '%s\n' 'unexpected dirty paths:'
  comm -23 \
    /tmp/rlt_closeout_dirty_paths.txt \
    /tmp/rlt_closeout_manifest_paths.txt
  exit 1
fi

printf 'review_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'manifest_sha256\t%s\n' "${expected_manifest_sha}"
printf 'manifest_files\t21\n'
printf 'visible_dirty_files\t%s\n' "$(wc -l < /tmp/rlt_closeout_dirty_paths.txt)"
printf 'pre_staged_files\t%s\n' "$(wc -l < /tmp/rlt_closeout_pre_staged_paths.txt)"
printf '%s\n' RLT_STAGE2_FRESH_CLOSEOUT_DEPLOY_REVIEW_OK
