import ast
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import sys

from checkpoint_store import archive_best, rotate_latest
from pour_protocol import standard_evaluation

def test_storage_and_schedule():
    from rlinf.utils.runner_utils import check_progress
    assert check_progress(1,200,5,5,1) == (False,False,False)
    assert check_progress(5,200,5,5,1) == (True,True,False)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for n in (1,2,3):
            d=root/'data'/f'global_step_{n}'
            d.mkdir(parents=True)
            (d/'dojo-complete.json').write_text('{}')
            (d/'optim').write_bytes(bytes([n]))
            rotate_latest(d)
            result=archive_best(d,root/'archive',root/'assets',n/10 if n<3 else 0,n) if (root/'assets').exists() else None
            if n==1:
                (root/'assets').mkdir()
                archive_best(d,root/'archive',root/'assets',.1,1)
        assert [x.name for x in (root/'data').iterdir()]==['global_step_3']
        assert result['round']==2 and (Path(result['checkpoint'])/'optim').read_bytes()==bytes([2])
        assert not (root/'archive/global_step_1').exists()

def test_native_pouring_accepts_a_stable_group_without_pose_screen():
    from native_entry import NativeLoop
    import inspect
    source=inspect.getsource(NativeLoop.reset)
    assert 'if not bad and not native_pour:' in source
    assert 'Liquid quantity differs' not in source
    assert 'invalid_reset_slots(self.env, self.n)' in source

