"""Validate new arrays with the accepted complete-video endpoint checks."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from observe import BUDGETS, derive


def validate(config):
    c = json.loads(Path(config).read_text())
    out = Path(c['output'])
    done = json.loads((out / 'done.json').read_text())
    episodes = json.loads((out / 'episodes.json').read_text())
    batch = c['num_envs']
    assert done['nstep_signals'] and len(episodes) == done['episodes'] == batch
    if c['acceptance']:
        assert json.loads((out / 'nstep-parity.json').read_text())['rng_exact']
    frames, rows = [], []
    for e in episodes:
        m = json.loads((out / e['frame_map']).read_text())
        ticks = [r['tick'] for r in m['frames']]
        assert ticks[0] == 0 and ticks[-1] == m['ticks'] == e['control_steps']
        assert len(ticks) == e['frames'] and all(0 < b-a <= m['stride'] for a,b in zip(ticks,ticks[1:]))
        cap = cv2.VideoCapture(str(out / e['video']))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert n == e['frames'] and n >= 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, n-1)
        ok, img = cap.read()
        cap.release()
        assert ok and img.size
        assert (out / f"episode_{e['slot']:02d}_control.jsonl").stat().st_size > 0
        frames.append(dict(slot=e['slot'], frames=n, terminal_frame_readable=True))
    for q in range(done['queries']):
        with np.load(out / f'query_{q:03d}.npz') as z:
            assert z['noise4_initial_noise'].shape == (batch,4,50,32)
            np.testing.assert_array_equal(z['noise4_initial_noise'][:,0], z['initial_noise_full'])
            np.testing.assert_array_equal(z['actions_env_n10'][:,0],z['env_action'])
            for n in BUDGETS:
                assert z[f'actions_model_n{n}'].shape == (batch,4,50,14)
                assert z[f'x_chain_n{n}'].shape == (batch,n+1,50,14)
                assert z[f'z_endpoint_n{n}'].shape == (batch,n,50,14)
                assert np.isfinite(z[f'actions_model_n{n}']).all()
                assert np.isfinite(z[f'actions_env_n{n}']).all()
                assert z[f'final_time_n{n}'] == 0
            for key,value in derive(z).items():
                np.testing.assert_array_equal(value,z[key])
            keys = [f'{kind}_n{n}' for n in BUDGETS for kind in ['noise4_relative','noise4_absolute','dv']]
            keys += [f'{kind}_{a}_{b}' for a,b in [(1,10),(1,5),(5,10)] for kind in ['u','rmse']]
            for slot in np.flatnonzero(z['active']):
                count = int(z['after'][slot] - z['before'][slot])
                assert 0 <= count <= 50
                for h in range(count):
                    row = dict(slot=int(slot),seed=episodes[slot]['actual'],query=q,
                               submitted_action=int(z['before'][slot])+h,chunk_position=h,
                               success=episodes[slot]['success'])
                    row.update({k:float(z[k][slot,h]) for k in keys})
                    rows.append(row)
        assert (out / f'query_{q:03d}_obs.npz').is_file()
    assert rows
    with (out / 'signals.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(passed=True,episodes=batch,successes=done['successes'],queries=done['queries'],
                submitted_actions=len(rows),frames=frames,
                video_validation='all coverage metadata and decoded final frames; unchanged accepted recorder',
                n1_dv='missing NaN, valid false',
                action_axis='submitted model prefix; physical frame alignment uses saved per-arm projection, not an exact TOPP inverse')
    (out / 'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    validate(sys.argv[1])
