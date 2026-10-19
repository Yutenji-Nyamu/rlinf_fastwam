"""Read the completed v2 smoke receipts once; no GPU work or reruns."""
import json
import time
from pathlib import Path

D = Path('/data/chenyiteng/deployment-20261019/pi05-noise-early-c20-sz1-v2')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert (D / 'smoke-complete.json').is_file()
    assert not (D / 'error.json').exists()
    assert not (D / 'smoke-acceptance.json').exists()
    result = {'time': time.time(), 'noise': [], 'policy': []}
    for g in (4, 5):
        accepted = read(D / 'noise-smoke' / f'acceptance-g{g}.json')
        assert accepted['validation']['passed'] and accepted['validation']['video_passed']
        c = read(accepted['config']); out = Path(c['output'])
        parity = read(out / 'noise4-parity.json')
        assert parity['main_action_exact'] and parity['rng_exact']
        result['noise'].append({'gpu': g, 'task': c['task'], 'output': str(out),
                                'done': read(out / 'done.json'), 'parity': parity,
                                'video_and_signal_validation': accepted['validation']})

        plan = read(D / 'switch-smoke' / f'lane-g{g}-plan.json')
        c = read(plan['stages'][0]['configs'][0]); out = Path(c['output'])
        done = read(out / 'done.json'); rows = read(out / 'decisions.json')
        episodes = read(out / 'episodes.json')
        assert done['episodes'] == len(episodes) == 32
        loaded = read(out / 'strong-loaded.json')
        assert loaded['strict'] and not loaded['missing_keys'] and not loaded['unexpected_keys']
        threshold = read(c['threshold_path'])['high']
        for row in rows:
            eligible = c['max_takeover_queries'] is None or row['query'] < 2
            assert row['strong'] == (eligible and row['u'] > threshold)
            assert row['eligible'] == eligible
            assert 0 <= row['start'] < row['end'] <= 200
            assert row['offered_actions'] == c['execute_chunk']
        later = [row for row in rows if row['query'] >= 2]
        assert later
        if g == 4:
            assert c['execute_chunk'] == 50 and c['max_takeover_queries'] == 2
            assert not any(row['strong'] for row in later)
        else:
            assert c['execute_chunk'] == 20 and c['max_takeover_queries'] is None
            assert all(row['start'] == row['query'] * 20 for row in rows)
        for episode in episodes:
            mapping = read(out / episode['frame_map'])
            assert mapping['frames'][0]['tick'] == 0
            assert mapping['frames'][-1]['tick'] == mapping['ticks'] == episode['control_steps']
            assert (out / episode['video']).stat().st_size > 0
        result['policy'].append({'gpu': g, 'condition': c['condition'], 'output': str(out),
                                 'done': done, 'strict_weight_load': loaded,
                                 'same_current_query_gate': True,
                                 'decisions': len(rows), 'strong': sum(r['strong'] for r in rows),
                                 'later_decisions': len(later), 'later_strong': sum(r['strong'] for r in later),
                                 'execute_chunk': c['execute_chunk'],
                                 'max_takeover_queries': c['max_takeover_queries'],
                                 'video_endpoints_complete': True})
    result['passed'] = True
    (D / 'smoke-acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
