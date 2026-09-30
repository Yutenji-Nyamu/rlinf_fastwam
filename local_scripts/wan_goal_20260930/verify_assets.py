"""Read-only CPU verification of the three pinned Wan Goal policy/WM assets.

Usage: python verify_assets.py --root /data/.../wan-goal-sz3 --output NEW_RECEIPT
Run after downloads finish. No downloads, imports of model code, or cache scans.
Exit 0 confirms required paths/sizes, seven large LFS hashes, and reset-file list.
Exit 2 means incomplete/mismatched evidence. Optional receipt is never overwritten.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def checked_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid manifest relative path: {relative}")
    return root.joinpath(*path.parts)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_repository(root, repo, hash_min_bytes):
    directory = checked_path(root, repo["local_subdir"])
    result = {"repo_id": repo["repo_id"], "revision": repo["revision"],
              "directory": str(directory), "required_files": repo["required_file_count"],
              "size_verified_files": 0, "size_verified_bytes": 0,
              "large_sha256": [], "errors": []}
    required = [entry for entry in repo["files"] if entry["required"]]
    if len(required) != repo["required_file_count"]:
        raise ValueError("Manifest required-file count mismatch")
    for entry in required:
        relative, expected_size = entry["path"], entry["size"]
        path = checked_path(directory, relative)
        try:
            if not path.is_file():
                raise FileNotFoundError("Required regular file absent")
            before = path.stat()
            if before.st_size != expected_size:
                raise ValueError(f"Size {before.st_size}, expected {expected_size}")
            result["size_verified_files"] += 1
            result["size_verified_bytes"] += expected_size
            if expected_size >= hash_min_bytes:
                expected_hash = entry.get("lfs_sha256")
                if not expected_hash or len(expected_hash) != 64:
                    raise ValueError("Large file lacks official LFS SHA-256 in manifest")
                print(f"Hashing {repo['local_subdir']}/{relative} ({expected_size} bytes)",
                      file=sys.stderr, flush=True)
                actual_hash = sha256_file(path)
                after = path.stat()
                stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                matched = actual_hash == expected_hash and stable
                result["large_sha256"].append({"path": relative, "bytes": expected_size,
                    "expected": expected_hash, "actual": actual_hash, "stable_during_read": stable,
                    "ok": matched})
                if not matched:
                    raise ValueError("LFS SHA-256 mismatch or file changed while hashing")
        except (OSError, ValueError) as exc:
            result["errors"].append({"path": relative, "error": str(exc)})

    if "dataset" in repo:
        expected = repo["dataset"]
        dataset = checked_path(directory, expected["relative_dir"])
        prefix = expected["relative_dir"] + "/"
        expected_names = {entry["path"][len(prefix):] for entry in required
                          if entry["path"].startswith(prefix) and entry["path"].endswith(".npy")}
        # Only the official reset directory itself is enumerated, never cache.
        try:
            actual_names = {p.name for p in dataset.iterdir() if p.is_file() and p.suffix == ".npy"}
            kir = sum(name.endswith("_kir.npy") for name in actual_names)
            counts = {"normal_npy": len(actual_names) - kir, "kir_npy": kir,
                      "total_npy": len(actual_names)}
            matches = (actual_names == expected_names and
                       all(counts[key] == expected[key] for key in counts))
            result["dataset"] = {"ok": matches, **counts,
                "missing_names": sorted(expected_names - actual_names),
                "unexpected_names": sorted(actual_names - expected_names)}
            if not matches:
                result["errors"].append({"path": expected["relative_dir"],
                                         "error": "Reset-file names or 496 normal / 246 KIR counts differ"})
        except OSError as exc:
            result["errors"].append({"path": expected["relative_dir"], "error": str(exc)})
    result["ok"] = not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Project root containing models/")
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("asset_manifest.json")))
    parser.add_argument("--output", help="Write a new receipt; existing files are never overwritten")
    args = parser.parse_args()
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "read_only": True, "repositories": [], "errors": []}
    try:
        root = Path(args.root).resolve(strict=True)
        manifest_path = Path(args.manifest).resolve(strict=True)
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        if manifest["schema_version"] != 1 or len(manifest["repositories"]) != 3:
            raise ValueError("Expected the three-repository schema-1 manifest")
        report.update(root=str(root), manifest_path=str(manifest_path),
                      manifest_sha256=hashlib.sha256(raw).hexdigest(),
                      hash_min_bytes=manifest["hash_min_bytes"], scope=manifest["scope"])
        for repo in manifest["repositories"]:
            report["repositories"].append(verify_repository(root, repo, manifest["hash_min_bytes"]))
        report["ok"] = all(repo["ok"] for repo in report["repositories"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["ok"] = False
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = "PINNED_ASSETS_VERIFIED" if report["ok"] else "INCOMPLETE_OR_MISMATCHED_ASSETS"
    report["limitation"] = "Files below 100 MB receive size checks only; this does not verify their full content or execute the model/reset data. External tokenizer and simulator assets are outside this three-repository receipt."
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        try:
            with Path(args.output).open("x", encoding="utf-8") as output:
                output.write(serialized + "\n")
        except OSError as exc:
            report["ok"] = False
            report["status"] = "INCOMPLETE_OR_MISMATCHED_ASSETS"
            report["errors"].append(f"Cannot write new receipt: {exc}")
            serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
