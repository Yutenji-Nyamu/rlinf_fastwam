"""Invoke the authorized private GeoWAM collector without vendoring its code."""

import argparse
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo", required=True)
    args, _ = parser.parse_known_args()
    script = Path(args.repo) / "scripts/geowam/appendix_signals.py"
    if not script.is_file():
        raise FileNotFoundError(
            "Prepare the private GeoWAM collector first: " + str(script)
        )
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
