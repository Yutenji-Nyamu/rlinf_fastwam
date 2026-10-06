"""Identity-recording DSRL/probe wrapper with only its own Ray-job cleanup."""
import argparse
import os
from pathlib import Path
import runpy
import resource
import signal
import sys
import time
from lease_common import checked, read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    p, b, op = checked(args.control)
    req = read(args.request)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(4096, hard)), hard))
    for path, digest in req['pins'].items():
        assert sha(path) == digest
    runtime = Path(req['runtime'])
    me = b.proc(os.getpid())
    save(runtime / 'driver-identity.json', {**me, 'namespace': req.get('namespace'),
                                          'request': str(args.request), 'time': time.time()}, True)
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(128 + sig))
    entry = Path(p['dsrl_repo']) / req['entry']
    if req['kind'] == 'probe':
        sys.argv = [str(entry), *req.get('args', [])]
        runpy.run_path(str(entry), run_name='__main__')
        return
    from rlinf.scheduler import Cluster
    Cluster.NAMESPACE = req['namespace']
    sys.argv = [str(entry), '--config-path', str(runtime), '--config-name', 'resolved',
                'hydra.run.dir=.', 'hydra.output_subdir=null', 'hydra.job.chdir=false',
                'hydra/job_logging=stdout']
    try:
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        import ray
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        try:
            op.cleanup_driver(p, {'namespace': req['namespace']}, runtime)
        finally:
            ray.shutdown()


if __name__ == '__main__':
    main()
