#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
SRC="$ROOT/RLinf/.venv"
DST="$ROOT/backups/RLinf-pi0-venv-golden-20260717"
EVIDENCE="$ROOT/fastwam-rlinf-setup/pi0-env-before-fastwam-20260717"

test -x "$SRC/bin/python"
test ! -e "$DST"
mkdir -p "$ROOT/backups" "$EVIDENCE"

"$SRC/bin/python" --version > "$EVIDENCE/python-version.txt" 2>&1
"$SRC/bin/python" -m pip freeze --all > "$EVIDENCE/pip-freeze-all.txt"
"$SRC/bin/python" -m pip list --format=json > "$EVIDENCE/pip-list.json"
"$SRC/bin/python" -m pip check > "$EVIDENCE/pip-check.txt" 2>&1 || true
grep -E '(^-e | @ file:|/root/autodl-tmp/)' \
  "$EVIDENCE/pip-freeze-all.txt" > "$EVIDENCE/editable-and-local.txt" || true
"$SRC/bin/python" - <<'PY' > "$EVIDENCE/key-versions.txt"
from importlib.metadata import PackageNotFoundError, version

import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
for package in (
    "rlinf",
    "ray",
    "transformers",
    "hydra-core",
    "omegaconf",
    "numpy",
    "sapien",
    "mplib",
    "nvidia_curobo",
    "warp-lang",
):
    try:
        print(package, version(package))
    except PackageNotFoundError:
        print(package, "ABSENT")
PY

rsync -aH --numeric-ids --info=stats1 "$SRC/" "$DST/"
rsync -aHn --delete --numeric-ids --itemize-changes \
  "$SRC/" "$DST/" > "$EVIDENCE/rsync-dry-run.txt"
test ! -s "$EVIDENCE/rsync-dry-run.txt"
du -sh "$SRC" "$DST" > "$EVIDENCE/du.txt"
printf '%s\n' "$SRC" > "$EVIDENCE/source-path.txt"
printf '%s\n' "$DST" > "$EVIDENCE/backup-path.txt"
touch "$EVIDENCE/backup-verified.done"
