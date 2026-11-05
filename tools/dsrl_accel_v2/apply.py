# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Apply the second acceleration patch to the exact previously accepted runtime."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(text: str) -> str:
    """Return the normalized source digest used by the manifest."""
    return hashlib.sha256(text.encode()).hexdigest()


def patch_text(source: str, patch: str) -> str:
    """Apply a unified patch with exact context checks and no fuzzy matching."""
    original = source.splitlines(keepends=True)
    out = []
    cursor = 0
    lines = patch.splitlines(keepends=True)
    index = 2
    while index < len(lines):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[index])
        assert match, lines[index]
        start = int(match[1]) - 1
        assert start >= cursor, "Overlapping patch hunks"
        out.extend(original[cursor:start])
        cursor = start
        index += 1
        while index < len(lines) and not lines[index].startswith("@@"):
            line = lines[index]
            index += 1
            if line.startswith("\\ No newline"):
                continue
            if line[0] in " -":
                assert original[cursor] == line[1:], (cursor, original[cursor], line)
                cursor += 1
            if line[0] in " +":
                out.append(line[1:])
    out.extend(original[cursor:])
    return "".join(out)


def apply(root: Path) -> None:
    """Validate every input before applying a patch or installing adapter code."""
    package = Path(__file__).resolve().parent
    root = root.resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    prepared = {}
    for rel, hashes in manifest.items():
        path = root / rel
        assert path.resolve().is_relative_to(root), rel
        before = path.read_text(encoding="utf-8")
        actual = digest(before)
        if actual == hashes["after"]:
            continue
        assert actual == hashes["before"], f"Source drift: {rel}: {actual}"
        patch = (package / "patches" / (rel.replace("/", "__") + ".patch")).read_text(
            encoding="utf-8"
        )
        after = patch_text(before, patch)
        assert digest(after) == hashes["after"], rel
        compile(after, rel, "exec")
        prepared[path] = after
    for path, after in prepared.items():
        temporary = path.with_suffix(".accel-v2.tmp")
        temporary.write_text(after, encoding="utf-8", newline="\n")
        temporary.replace(path)
    destination = root / "tools/dsrl_accel_v2"
    if destination.resolve() != package:
        destination.mkdir(parents=True, exist_ok=True)
        for source in package.iterdir():
            if source.is_file() and source.suffix in (".py", ".json", ".md"):
                target = destination / source.name
                temporary = target.with_suffix(target.suffix + ".install.tmp")
                temporary.write_bytes(source.read_bytes())
                temporary.replace(target)
    print(
        json.dumps(
            {"patched": len(prepared), "verified": len(manifest), "root": str(root)}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    apply(parser.parse_args().root)
