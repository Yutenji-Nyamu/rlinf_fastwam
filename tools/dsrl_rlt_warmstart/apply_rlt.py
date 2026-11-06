# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Apply audited patches only to the exact recorded exported DSRL source."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def patch_text(source, patch):
    original = source.splitlines(keepends=True)
    out = []
    cursor = 0
    lines = patch.splitlines(keepends=True)
    index = 2
    while index < len(lines):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[index])
        assert match, lines[index]
        start = int(match[1]) - 1
        assert start >= cursor
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


def apply(root):
    package = Path(__file__).resolve().parent
    root = root.resolve()
    manifest = json.loads((package / "rlt-manifest.json").read_text(encoding="utf-8"))
    prepared = {}
    for rel, hashes in manifest.items():
        path = root / rel
        assert path.resolve().is_relative_to(root)
        before = path.read_text(encoding="utf-8")
        actual = digest(before)
        if actual == hashes["after"]:
            continue
        assert actual == hashes["before"], f"Source drift: {rel}: {actual}"
        patch = (
            package / "rlt-patches" / (rel.replace("/", "__") + ".patch")
        ).read_text(encoding="utf-8")
        after = patch_text(before, patch)
        assert digest(after) == hashes["after"], rel
        compile(after, rel, "exec")
        prepared[path] = after
    for path, after in prepared.items():
        temporary = path.with_suffix(".dsrl-patch.tmp")
        temporary.write_text(after, encoding="utf-8", newline="\n")
        temporary.replace(path)
    destination = root / "tools/dsrl_rlt_warmstart"
    if destination.resolve() != package:
        destination.mkdir(exist_ok=True)
        for path in package.glob("*.py"):
            shutil.copyfile(path, destination / path.name)
    print(
        json.dumps(
            {"patched": len(prepared), "verified": len(manifest), "root": str(root)}
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    apply(p.parse_args().root)
