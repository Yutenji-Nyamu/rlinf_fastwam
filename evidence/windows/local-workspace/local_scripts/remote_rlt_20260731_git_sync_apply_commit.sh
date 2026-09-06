#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
staging=/root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/staging
expected_head=46a2d19bae629eaa57830f5faeac71ac81a1a494
paths=(
  HANDOFF.md
  docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
  docs/rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
)

cd "$repo"
[[ "$(git branch --show-current)" == "codex/rlt-pi0-robotwin" ]]
[[ "$(git rev-parse HEAD)" == "$expected_head" ]]
mapfile -t preexisting_paths < <(
  {
    git diff --name-only
    git ls-files --others --exclude-standard
  } | sort -u
)
for candidate in "${preexisting_paths[@]}"; do
  allowed=0
  for path in "${paths[@]}"; do
    [[ "$candidate" == "$path" ]] && allowed=1
  done
  [[ "$allowed" == 1 ]] || {
    printf 'unexpected pre-existing worktree path: %s\n' "$candidate" >&2
    exit 1
  }
done

cat >"$staging/expected_sha256.txt" <<'EOF'
99bead7498eb8d8ae3ce4e97f1dd0f14add08825dcab214eea22078e2c722c97  HANDOFF.md
692e91a5bcb4b67445b2b79587835642e273a48f6491e2e47a685619eec8b094  docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
8088b6c1d5a2a34e5033293b315c5d56fe4aecce4591073570a82aeba09b5db0  docs/rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md
f39fca5acce24219aa46b418e83f7044a4eb2ba8a9c711170aae2da0f6aa69f6  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
EOF
(
  cd "$staging"
  sha256sum -c expected_sha256.txt
)

for path in "${paths[@]}"; do
  install -D -m 0644 "$staging/$path" "$repo/$path"
done

mapfile -t actual_paths < <(
  {
    git diff --name-only
    git ls-files --others --exclude-standard
  } | sort -u
)
if [[ "${actual_paths[*]}" != "${paths[*]}" ]]; then
  printf 'unexpected worktree paths:\n' >&2
  printf '%s\n' "${actual_paths[@]}" >&2
  exit 1
fi

python - "${paths[@]}" <<'PY'
from pathlib import Path
import sys

for raw in sys.argv[1:]:
    path = Path(raw)
    payload = path.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise SystemExit(f"UTF-8 BOM is forbidden: {path}")
    text = payload.decode("utf-8")
    if not text.endswith("\n"):
        raise SystemExit(f"missing final newline: {path}")
    trailing = [
        number
        for number, line in enumerate(text.splitlines(), 1)
        if line.endswith((" ", "\t"))
    ]
    if trailing:
        raise SystemExit(f"trailing whitespace in {path}: {trailing[:10]}")
    import re
    if len(re.findall(r"(?m)^```", text)) % 2:
        raise SystemExit(f"odd Markdown fence count: {path}")
    print(f"MARKDOWN_OK {path}")
PY

git add -- "${paths[@]}"
git diff --cached --check
echo "CACHED_STATUS_BEGIN"
git status --short
echo "CACHED_STATUS_END"
echo "CACHED_DIFFSTAT_BEGIN"
git diff --cached --stat
echo "CACHED_DIFFSTAT_END"
echo "CACHED_NAMES_BEGIN"
git diff --cached --name-status
echo "CACHED_NAMES_END"

git commit -m "docs(rlt): record cycle 480 formal closeout"

echo "POST_COMMIT_BEGIN"
git rev-parse HEAD
git log -4 --oneline --decorate
git status --short --branch
git rev-list --left-right --count '@{upstream}...HEAD'
echo "POST_COMMIT_END"
