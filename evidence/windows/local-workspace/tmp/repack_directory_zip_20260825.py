from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, root.name / path.relative_to(root))
    with zipfile.ZipFile(args.zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        print(f"entries={len(archive.infolist())} bytes={args.zip_path.stat().st_size}")


if __name__ == "__main__":
    main()
