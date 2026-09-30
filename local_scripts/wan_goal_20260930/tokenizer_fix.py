"""Verify the pinned OpenPI tokenizer; copy the installer copy only if absent.

No downloads, package changes, source edits, or replacement of existing files.
prepare_pi05_env.sh already snapshots the official repository into
OPENPI_DATA_HOME, which normally makes this verification-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path("/data/chenyiteng/projects/wan-goal-sz3")
REVISION = "befaa248e4f82954b625a421658f933dfd1a97a0"
SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
SIZE = 4264023
RELATIVE = Path("big_vision/paligemma_tokenizer.model")


def verify(path: Path) -> bytes:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if len(data) != SIZE or actual != SHA256:
        raise ValueError(f"Tokenizer content mismatch at {path}: bytes={len(data)}, sha256={actual}; preserving file")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("OPENPI_DATA_HOME", str(ROOT / "models/openpi-assets"))))
    parser.add_argument("--source", type=Path,
                        default=Path(os.environ.get("DOWNLOAD_DIR", str(ROOT / "assets"))) / ".cache/openpi" / RELATIVE)
    parser.add_argument("--check", action="store_true", help="Verify or describe the missing-target copy without writing")
    parser.add_argument("--receipt", type=Path, help="Optional new JSON receipt; never overwrite an existing receipt")
    args = parser.parse_args()
    cache = args.cache_dir.expanduser().resolve()
    target = cache / RELATIVE
    result = {"repo": "RLinf/openpi_tokenizer", "revision": REVISION,
              "file": RELATIVE.as_posix(), "sha256": SHA256, "bytes": SIZE,
              "cache_dir": str(cache), "target": str(target), "downloaded": False}

    if target.exists() or target.is_symlink():
        verify(target)
        result["status"] = "EXISTING_PINNED_TOKENIZER_VERIFIED"
    else:
        source = args.source.expanduser().resolve(strict=True)
        data = verify(source)
        result["source"] = str(source)
        if args.check:
            result["status"] = "MISSING_TARGET_PINNED_FALLBACK_AVAILABLE"
        else:
            # Refuse a nested cache symlink redirect; create only this cache entry.
            target.parent.resolve().relative_to(cache)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.parent.resolve(strict=True).relative_to(cache)
            descriptor, temporary = tempfile.mkstemp(prefix=".paligemma-verified-", dir=target.parent)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.chmod(temporary, 0o644)
                # Hard-link publication is atomic and fails if another writer wins.
                # Unlike replace(), this can never overwrite an existing tokenizer.
                try:
                    os.link(temporary, target)
                    result["status"] = "COPIED_PINNED_INSTALLER_TOKENIZER"
                except FileExistsError:
                    result["status"] = "CONCURRENT_PINNED_TOKENIZER_VERIFIED"
                verify(target)
            finally:
                Path(temporary).unlink(missing_ok=True)

    payload = json.dumps(result, indent=2) + "\n"
    if args.receipt:
        with args.receipt.open("x", encoding="utf-8") as output:
            output.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
