"""Read-only live frozen-source audit. A candidate manifest is never auto-trusted."""
import argparse
import json
from pathlib import Path
from common import HERE, account, exclusive, own_path, read, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--write-candidate', type=Path)
    args = parser.parse_args()
    account(); base = own_path(args.source_dir)
    expected = read(HERE / 'base-source-sha256.json')
    actual = {name: sha(own_path(base / name)) for name in expected}
    changed = {name: dict(expected=expected[name], actual=actual[name])
               for name in expected if expected[name] != actual[name]}
    if args.write_candidate:
        target = own_path(args.write_candidate, exists=False)
        assert target != HERE / 'base-source-sha256.json'
        exclusive(target, actual)
    print(json.dumps(dict(source_dir=str(base), resolved_source_dir=str(base.resolve()),
                          matches_reviewed_copy=not changed, changed=changed, actual=actual)))
    raise SystemExit(1 if changed else 0)


if __name__ == '__main__':
    main()
