"""Paired evaluation schedule, shared by the worker and offline tests."""

def episode_schedule(seeds, repeats):
    if not seeds or len(set(seeds)) != len(seeds) or repeats < 1:
        raise ValueError('Expected distinct scene seeds and positive repeats')
    return [dict(episode_id=r*len(seeds)+i, scene_seed=int(seed), repeat=r)
            for r in range(repeats) for i, seed in enumerate(seeds)]


def batches(episodes, parallel):
    if parallel < 1:
        raise ValueError('parallel must be positive')
    return [episodes[i:i+parallel] for i in range(0, len(episodes), parallel)]


def summarize(rows):
    if not rows:
        return dict(episodes=0, successes=0, success_rate=None)
    if len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate episode results')
    n = sum(bool(r['success']) for r in rows)
    return dict(episodes=len(rows), successes=n, success_rate=n/len(rows))


def paired_comparison(base, teacher):
    a={x['episode_id']:x for x in base}; b={x['episode_id']:x for x in teacher}
    if a.keys()!=b.keys():
        raise ValueError('Base/teacher episode sets differ')
    wins=losses=0
    for k in a:
        for key in ['scene_seed','repeat','accepted_seed','instruction','chunk_steps']:
            if a[k][key]!=b[k][key]:
                raise ValueError(f'Pair {k}: {key} differs')
        wins += bool(b[k]['success']) and not a[k]['success']
        losses += bool(a[k]['success']) and not b[k]['success']
    return dict(base=summarize(base), teacher=summarize(teacher),
                teacher_only_success=wins, base_only_success=losses,
                difference=(wins-losses)/len(a))
