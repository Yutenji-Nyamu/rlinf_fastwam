"""CPU-only wait, then reuse the accepted owner for one full-state continuation."""
import os
import signal
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, '/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops')
from common import proc, read, same, save


def checkpoint_ready(path):
    path = Path(path)
    return all((path / name).is_file() for name in (
        'dojo-complete.json',
        'actor/local_shard_checkpoint/checkpoint_rank_0.pt',
        'actor/sac_components/dsrl_trainer_state_rank_0.pt',
    ))


def main(directory):
    d = Path(directory)
    p = read(d / 'continuation.json')
    me = proc(os.getpid())
    save(d / 'identity.json', me)
    save(d / 'status.json', dict(phase='WAIT_CURRENT_DSRL', time=time.time(), target=400))
    def stop(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not (d / 'registered.json').exists():
            time.sleep(.2)
        while same(p['predecessor']):
            time.sleep(.5)
        assert Path(p['previous_release']).is_file(), 'Previous owner has not released'
        assert checkpoint_ready(p['checkpoint']), 'Final full checkpoint unavailable; return RLT'
        # Carry forward the accepted simulator's already rejected layouts.
        if p['kind'] == 'dojo':
            target = Path(p['deployment']) / f"native-g{p['gpu']}.json"
            native = read(target)
            invalid = {k: set(v) for k, v in native.get('invalid_layouts', {}).items()}
            log = Path(p['previous_deployment']) / f"native-g{p['gpu']}/invalid-resets.jsonl"
            import json
            for line in log.read_text().splitlines() if log.exists() else []:
                row = json.loads(line)
                invalid.setdefault(row['mode'], set()).update(row['invalid_layouts'])
            native['invalid_layouts'] = {k: sorted(v) for k, v in invalid.items()}
            save(target, native)
        save(d / 'continuation-ready.json', dict(time=time.time(), checkpoint=p['checkpoint'], target=400))
        # Same process identity/reservation throughout; no transient RLT launch.
        save(d / 'registered.json', dict(identity=proc(os.getpid()), time=time.time()))
        if p['kind'] == 'dojo':
            from tools.dojo_offpolicy.owner import main as owner
        else:
            from tools.robotwin_dsrl_full.queue import main as owner
        owner(str(d))
    except BaseException:
        if not (d / 'released.json').exists():
            result = dict(phase='FAILED_RETURN_RLT', time=time.time(), error=traceback.format_exc())
            save(d / 'status.json', result)
            save(d / 'released.json', result)
        raise


if __name__ == '__main__':
    main(sys.argv[1])
