"""Allow only independently evidenced natural completion of the old clean BC."""
import importlib.util
import json
from pathlib import Path

stage = Path('/data/chenyiteng/results/server-maintenance-20260908/bc-dvac-new')
spec = importlib.util.spec_from_file_location('bc_new_smoke_ops', stage / 'ops.py')
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)
bc_run = Path('/home/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc4x1-b1024-u5-m10-seed42-eval8x4-gpu6-formal100-20260908-v1')
allowed = {1786521: 173171171, 1786527: 173171177, 1786531: 173171184}
natural = {}


def check_protected(protected):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ended = []
    for pid, expected in protected.items():
        try:
            actual = ops.proc(pid)
        except FileNotFoundError:
            assert pid in allowed and expected['uid'] == 1003 and expected['start'] == allowed[pid], expected
            ended.append(expected)
        else:
            assert all(actual[key] == expected[key] for key in ('pid', 'uid', 'start')), (expected, actual)
    if ended:
        assert (bc_run / 'exit_code.txt').read_text().strip() == '0'
        assert ops.git(ops.ROOT.parent / 'pi05-online-bc-u10', 'rev-parse', 'HEAD') == '01d770db3988da7862454e97434d4ff08f726fa2'
        ea = EventAccumulator(str(bc_run / 'tensorboard'), size_guidance={'scalars': 0})
        ea.Reload()
        collected = ea.Scalars('env/success_once')
        trained = ea.Scalars('train/bc/actor_loss')
        evaluated = ea.Scalars('eval/success_once')
        assert len(collected) == 100 and max(x.step for x in collected) == 99
        assert max(x.step for x in trained) == 99 and max(x.step for x in evaluated) == 99
        natural.update(run=str(bc_run), exit_code=0, completed_rounds=100,
                       final_fixed32=round(evaluated[-1].value * 32),
                       terminated_processes=ended,
                       finished_at=(bc_run / 'finished_at.txt').read_text().strip())


ops.assert_processes_unchanged = check_protected
ops.verify()
path = stage / 'smoke-verification.json'
result = json.loads(path.read_text())
result.pop('protected_unchanged')
result.update(protected_verified=True, natural_completed_bc=natural,
              protected_policy='Unchanged PID/UID/start, except exact old clean BC processes with verified original R100/exit0 completion.')
path.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
