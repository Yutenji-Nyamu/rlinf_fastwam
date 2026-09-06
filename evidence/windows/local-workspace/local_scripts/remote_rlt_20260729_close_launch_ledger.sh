#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
ledger=docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
expected_parent=d7c3ca7e2ddfc8d0b3c376ec6d30ba89b965a5dc

cd "$repo"
test "$(git rev-parse HEAD)" = "$expected_parent"
test "$(git status --porcelain)" = " M $ledger"
git diff --check -- "$ledger"

/root/autodl-tmp/RLinf/.venv/bin/python -B - "$ledger" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
assert "## 53. A053：文档收口、提交与有限网络重试" in text
assert "d7c3ca7e2ddfc8d0b3c376ec6d30ba89b965a5dc" in text
print(f"LEDGER_CLOSE_REVIEW_OK bytes={path.stat().st_size}")
PY

git add -- "$ledger"
test "$(git diff --cached --name-only)" = "$ledger"
git diff --cached --check
git commit -m "docs(rlt): close formal launch ledger"

timeout 240s git push personal codex/rlt-pi0-robotwin

test -z "$(git status --porcelain)"
test "$(git rev-list --left-right --count '@{upstream}...HEAD')" = $'0\t0'
head=$(git rev-parse HEAD)
remote=$(git ls-remote personal refs/heads/codex/rlt-pi0-robotwin | awk '{print $1}')
test "$head" = "$remote"
printf 'HEAD=%s\nREMOTE=%s\nSTATUS=clean\nUPSTREAM=0/0\n' "$head" "$remote"
