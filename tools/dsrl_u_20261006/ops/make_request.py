"""Freeze one source/config request; no model import, GPU or launch."""
import argparse
from pathlib import Path
import subprocess
import time
from lease_common import checked, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--gpu', type=int, choices=(6, 7), required=True)
    parser.add_argument('--kind', choices=('probe', 'smoke', 'formal'), required=True)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--entry', default='examples/embodiment/train_embodied_agent.py')
    parser.add_argument('--namespace')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--probe-arg', action='append', default=[])
    args = parser.parse_args()
    p, b, op = checked(args.control)
    repo = Path(p['dsrl_repo'])
    paths = [repo / args.entry, *sorted((repo / 'rlinf').rglob('*.py')),
             *(repo / relative for relative in args.source)]
    if args.kind != 'probe':
        paths.append(args.runtime / 'resolved.yaml')
        assert args.namespace and not args.probe_arg
    else:
        assert args.namespace is None
    for path in paths:
        assert path.is_file()
    req = {'request_id': args.request_id, 'gpu': args.gpu, 'kind': args.kind,
           'runtime': str(args.runtime), 'entry': args.entry, 'namespace': args.namespace,
           'args': args.probe_arg, 'env': {}, 'pins': {str(path): sha(path) for path in paths},
           'created_at': time.time(),
           'repo_head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()}
    save(args.output, req, True)
    print(__import__('json').dumps({'request': str(args.output), 'sha256': sha(args.output), 'launched': False}))


if __name__ == '__main__':
    main()
