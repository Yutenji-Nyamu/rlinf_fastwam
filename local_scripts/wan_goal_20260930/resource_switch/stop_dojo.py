"""Request the current owned Dojo controller to exit; outer retains restoration."""
import argparse
import os
from pathlib import Path
import signal
import time
from common import account, exclusive, load_base, own_path, pidfd_open, pidfd_send, read, sha


def signal_bound(run, bound, sig):
    # Same ownership rechecks as the frozen watchdog; compatibility changes only
    # pidfd access when this Python build omitted its Linux native bindings.
    from hang_watchdog import bound_run, identity
    fresh = bound_run(run)
    assert fresh is not None and fresh['parent'] == bound['parent'] and fresh['controller'] == bound['controller']
    fd = pidfd_open(bound['controller']['pid'])
    try:
        assert identity(bound['controller']['pid'], bound['controller']['start']) == bound['controller']
        pidfd_send(fd, sig)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-source-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--intent', type=Path, required=True)
    parser.add_argument('--reason', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    account(); base, _ = load_base(args.base_source_dir)
    from hang_watchdog import bound_run
    # The frozen watchdog compares its resolved attempt.parent to this argument.
    # Resolve only at that API boundary; other launch paths retain /data aliases.
    run = own_path(args.run_dir).resolve()
    bound = bound_run(run)
    assert bound, 'Current run is not an exactly bound EVALUATING controller'
    intent = own_path(args.intent, exists=False)
    assert intent.parent.resolve() == Path(bound['attempt']).resolve() and not intent.exists()
    result = dict(time=time.time(), action='SIGTERM_CURRENT_DOJO_CONTROLLER',
                  reason=args.reason, bound=bound, execute=args.execute,
                  active_pointer=read(run / 'active-continuation.json'),
                  base_watchdog_sha256=sha(base / 'hang_watchdog.py'))
    if args.execute:
        exclusive(intent, result)
        signal_bound(run, bound, signal.SIGTERM)
    print(__import__('json').dumps(result))


if __name__ == '__main__':
    main()
