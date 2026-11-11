"""Invoke the authorized private GeoWAM collector without vendoring its code."""

import argparse
import json
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args, _ = parser.parse_known_args()
    ready = Path(args.manifest).parent / "geowam-data-ready.json"
    if not ready.exists() or json.loads(ready.read_text()).get("state") != "complete":
        from .common import write_json

        write_json(
            Path(args.output) / "skipped.json", {"state": "skipped_download_not_ready"}
        )
        print(
            "GeoWAM inputs are not ready at its turn; skipping to original training.",
            flush=True,
        )
        return
    script = Path(args.repo) / "scripts/geowam/appendix_signals.py"
    if not script.is_file():
        raise FileNotFoundError(
            "Prepare the private GeoWAM collector first: " + str(script)
        )
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
