#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/tmp/rlt_stage2_fresh_closeout_20260730_v4
expected_head=9bb2dd78feff7133780c3df6a88618d10168c4e4
handoff_sha=b0002425bc4c82dc69753037f01126e0c5a7d3e7b30c7465c191ac61a0565c79
ledger_sha=71b4c8da227eb13db71074563dedb516f45cfeda544098449a86d3f69b411806

cd "${repo}"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = "${expected_head}"
test -z "$(git status --short)"
read -r left right < <(git rev-list --left-right --count '@{upstream}...HEAD')
test "${left}" = 0
test "${right}" = 0
test "$(sha256sum "${stage}/HANDOFF.md" | cut -d' ' -f1)" = "${handoff_sha}"
test "$(
  sha256sum \
    "${stage}/docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md" \
    | cut -d' ' -f1
)" = "${ledger_sha}"

install -D -m 0644 "${stage}/HANDOFF.md" "${repo}/HANDOFF.md"
install -D -m 0644 \
  "${stage}/docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md" \
  "${repo}/docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md"

/root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
from pathlib import Path
import re

repo = Path("/root/autodl-tmp/RLinf_rlt_pi0_robotwin")
paths = [
    repo / "HANDOFF.md",
    repo / (
        "docs/rlinf-robotwin-pi0-rltoken/evidence/"
        "IMPLEMENTATION_LOG.md"
    ),
]
for path in paths:
    text = path.read_bytes().decode("utf-8", errors="strict")
    for number, line in enumerate(text.splitlines(), 1):
        assert line.rstrip(" \t") == line, f"trailing whitespace {path}:{number}"
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        assert (path.parent / target).resolve().exists(), (
            f"missing link {path}:{target}"
        )
PY

git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
mapfile -t staged < <(git diff --cached --name-only | LC_ALL=C sort)
test "${#staged[@]}" = 2
test "${staged[0]}" = HANDOFF.md
test "${staged[1]}" = \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
git diff --cached --check
git commit -m "docs(rlt): close Stage 2 fresh handoff"
new_head="$(git rev-parse HEAD)"
test -z "$(git status --short)"

main_code="$(
  curl -L -sS -o /dev/null \
    --connect-timeout 7 --max-time 10 \
    -w '%{http_code}' https://github.com || true
)"
test "${main_code}" = 200
GIT_TERMINAL_PROMPT=0 timeout 60 \
  git push personal HEAD:codex/rlt-pi0-robotwin
remote_head="$(
  timeout 15 git ls-remote \
    personal refs/heads/codex/rlt-pi0-robotwin \
    | awk '{print $1}'
)"
test "${remote_head}" = "${new_head}"
read -r left_after right_after < <(
  git rev-list --left-right --count '@{upstream}...HEAD'
)
test "${left_after}" = 0
test "${right_after}" = 0

printf 'closeout_commit\t%s\n' "${new_head}"
printf 'remote_head\t%s\n' "${remote_head}"
printf 'left_right_after\t%s/%s\n' "${left_after}" "${right_after}"
printf 'dirty_count\t0\n'
printf '%s\n' RLT_STAGE2_TERMINAL_LEDGER_CLOSEOUT_OK
