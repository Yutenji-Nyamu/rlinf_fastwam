"""Archive exact small configurations and completed smoke records for plotting."""
import json
import tarfile
from pathlib import Path

D = Path('/data/chenyiteng/deployment-20261019/pi05-noise-early-c20-sz1-v2')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert read(D / 'smoke-acceptance.json')['passed']
    target = D / 'inference-smoke-and-configs.tar.gz'
    assert not target.exists()
    prepared = read(D / 'prepared.json')
    files = {}
    for name in ('prepared.json', 'early-calibration.json', 'metadata-revision.json',
                 'smoke-complete.json', 'smoke-acceptance.json', 'cutover-complete.json',
                 'noise-signal-stats.json'):
        files[D / name] = 'transaction/' + name
    for stage in prepared['phase_order']:
        for path in (D / stage).glob('lane-g*-plan.json'):
            files[path] = str(Path('transaction') / stage / path.name)
    for key, root in prepared['roots'].items():
        root = Path(root)
        if not root.exists():
            continue
        for directory in ('configs', 'thresholds'):
            for path in (root / directory).glob('*.json'):
                files[path] = str(Path(key) / directory / path.name)
        for name in ('manifest.json', 'frozen-thresholds.json'):
            if (root / name).exists():
                files[root / name] = str(Path(key) / name)
        if key in ('noise', 'smoke'):
            for done in (root / 'batches').glob('*/done.json'):
                for path in done.parent.glob('*.json'):
                    files[path] = str(Path(key) / 'batches' / done.parent.name / path.name)
    with tarfile.open(target, 'x:gz') as archive:
        for path, name in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
    result = {'path': str(target), 'bytes': target.stat().st_size, 'files': len(files),
              'contents': 'Exact configs/seeds/thresholds, completed smoke JSON including frame maps; no environment dumps, weights, videos or raw NPZ/control JSONL.'}
    (D / 'evidence-archive.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
