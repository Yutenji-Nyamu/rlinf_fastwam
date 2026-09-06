from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path


WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_stop_g51_20260822"
)
REPORT = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/"
    "17_R_ONLY_G51_CLOSEOUT_AND_V3_RANGE_DISCUSSION_20260822.md"
)
ZIP_PATH = WORKSPACE / "exports/idea2_dvac_v2_r_only_formal_stop_g51_20260822.zip"
ROOT = "idea2_dvac_v2_r_only_formal_stop_g51_20260822"


def add_tree(selected: list[tuple[Path, str]], local_root: Path, archive_root: str) -> None:
    for path in sorted(local_root.rglob("*")):
        if path.is_file():
            selected.append((path, f"{archive_root}/{path.relative_to(local_root).as_posix()}"))


selected: list[tuple[Path, str]] = []
for name in (
    "README.md",
    "OPERATION_LEDGER.md",
    "FINAL_READONLY_AUDIT.stdout.txt",
    "FINAL_READONLY_AUDIT.stderr.txt",
    "FINAL_READONLY_AUDIT.exitcode.txt",
    "DOWNLOAD_SKIPPED.txt",
    "analyze_closeout_g51.py",
    "analyze_v3_range_counterfactual_g51.py",
):
    selected.append((EVIDENCE / name, name))
selected.append((REPORT, "REPORT.md"))

add_tree(selected, EVIDENCE / "analysis", "analysis")
add_tree(selected, EVIDENCE / "raw/run/dvac_train", "raw/run/dvac_train")
add_tree(selected, EVIDENCE / "raw/run/control_trace", "raw/run/control_trace")
selected.append((EVIDENCE / "raw/run/metrics.log", "raw/run/metrics.log"))

for path in sorted((EVIDENCE / "raw/runtime").iterdir()):
    if path.is_file():
        selected.append((path, f"raw/runtime/{path.name}"))

missing = [str(path) for path, _ in selected if not path.is_file()]
if missing:
    raise FileNotFoundError("missing package files:\n" + "\n".join(missing))

rows: list[dict[str, str | int]] = []
for path, archive_name in selected:
    payload = path.read_bytes()
    rows.append(
        {
            "archive_path": archive_name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )

manifest_io = io.StringIO(newline="")
writer = csv.DictWriter(manifest_io, fieldnames=["archive_path", "bytes", "sha256"])
writer.writeheader()
writer.writerows(rows)
manifest = manifest_io.getvalue().encode("utf-8")
sha_lines = "".join(f"{row['sha256']}  {row['archive_path']}\n" for row in rows).encode("utf-8")
(EVIDENCE / "FILE_MANIFEST.csv").write_bytes(manifest)
(EVIDENCE / "SHA256SUMS.txt").write_bytes(sha_lines)

ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path, archive_name in selected:
        archive.write(path, f"{ROOT}/{archive_name}")
    archive.writestr(f"{ROOT}/FILE_MANIFEST.csv", manifest)
    archive.writestr(f"{ROOT}/SHA256SUMS.txt", sha_lines)

with zipfile.ZipFile(ZIP_PATH) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP CRC failed at {bad}")
    count = len(archive.infolist())

print(f"ZIP={ZIP_PATH}")
print(f"BYTES={ZIP_PATH.stat().st_size}")
print(f"FILES={count}")
print(f"SHA256={hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()}")
