#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
patch_path=/tmp/ogpo_precommit_20260807.patch
task_index=/tmp/ogpo_precommit_index_$$
trap 'rm -f "$task_index"' EXIT

expected=(
  examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml
  examples/embodiment/train_embodied_agent.py
  rlinf/algorithms/ogpo/__init__.py
  rlinf/algorithms/ogpo/core.py
  rlinf/config.py
  rlinf/data/ogpo_replay.py
  rlinf/models/embodiment/base_policy.py
  rlinf/models/embodiment/modules/ogpo_critic.py
  rlinf/models/embodiment/modules/ogpo_modules.py
  rlinf/models/embodiment/openpi/__init__.py
  rlinf/models/embodiment/openpi/openpi_action_model.py
  rlinf/models/embodiment/openpi/openpi_ogpo.py
  rlinf/runners/embodied_runner.py
  rlinf/workers/actor/fsdp_ogpo_policy_worker.py
  rlinf/workers/env/env_worker.py
  rlinf/workers/rollout/hf/huggingface_worker.py
  tests/algorithms/test_ogpo_core.py
  tests/data/test_ogpo_replay.py
  tests/embodiment/ogpo_fsdp_ema_fixture.py
  tests/embodiment/ogpo_real_fsdp_ema_probe.py
  tests/embodiment/ogpo_real_fsdp_update_probe.py
  tests/embodiment/test_ogpo_critic.py
  tests/embodiment/test_openpi_ogpo_adapter.py
  tests/workers/test_ogpo_checkpoint_sidecar.py
  tests/workers/test_ogpo_env_trace.py
  tests/workers/test_ogpo_row_schedule.py
)

test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = \
  codex/ogpo-pi0-robotwin
test -z "$(git -C "$repo" diff --cached --name-only)"
test ! -e "$patch_path"

REPO="$repo" EXPECTED="$(printf '%s\n' "${expected[@]}")" \
  /root/autodl-tmp/RLinf/.venv/bin/python - <<'PY'
import os
import subprocess

repo = os.environ["REPO"]
expected = sorted(os.environ["EXPECTED"].splitlines())
status = subprocess.run(
    ["git", "-C", repo, "status", "--porcelain=v1", "--untracked-files=all"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
actual = sorted(line[3:] for line in status)
if actual != expected:
    raise SystemExit(f"unexpected precommit paths: actual={actual} expected={expected}")
print(f"PRECOMMIT_SCOPE_OK files={len(actual)}")
PY

GIT_INDEX_FILE="$task_index" git -C "$repo" read-tree HEAD
GIT_INDEX_FILE="$task_index" git -C "$repo" add -- "${expected[@]}"
GIT_INDEX_FILE="$task_index" git -C "$repo" diff --cached --check HEAD
GIT_INDEX_FILE="$task_index" git -C "$repo" diff \
  --cached --binary --full-index HEAD > "$patch_path"

printf 'FULL_PATCH_NAME_STATUS\n'
GIT_INDEX_FILE="$task_index" git -C "$repo" diff --cached --name-status HEAD
printf 'FULL_PATCH_STAT\n'
GIT_INDEX_FILE="$task_index" git -C "$repo" diff --cached --stat HEAD
printf 'FULL_PATCH_NUMSTAT\n'
GIT_INDEX_FILE="$task_index" git -C "$repo" diff --cached --numstat HEAD
printf 'FULL_PATCH_ARTIFACT\n'
wc -c "$patch_path"
sha256sum "$patch_path"

PATCH_PATH="$patch_path" /root/autodl-tmp/RLinf/.venv/bin/python - <<'PY'
import os
import re

path = os.environ["PATCH_PATH"]
patterns = {
    "private-key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "credential-url": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
    "named-secret": re.compile(
        r"(?i)\b(?:password|passwd|secret|token|api[_-]?key|access[_-]?key)\b"
        r"\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
    "common-token": re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"),
}
matches = []
with open(path, encoding="utf-8", errors="replace") as handle:
    for number, line in enumerate(handle, 1):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for label, pattern in patterns.items():
            if pattern.search(line):
                matches.append((number, label))
if matches:
    raise SystemExit(f"SECRET_SCAN_MATCH metadata={matches}")
print("SECRET_SCAN_OK")
PY
