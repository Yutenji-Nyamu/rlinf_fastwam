"""Freeze two Clean CP140 -> U tau3 CP70 C20 sweeps using accepted configs."""
import copy
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

from calibration_pool import collect, deduplicate
from protocol import read, save, validate_config
from threshold_sweep import conditions

R = Path(__file__).resolve().parents[2]
BASE = Path('/data/chenyiteng')
OLD = BASE / 'results/pi05-policy-eval/clean-weak-c20-20261020-v1'
BUNDLE = BASE / 'exports/clean-c20-followups-20261021'
WEAK = BASE / 'models/protected-eval/pillbottle/clean_c17_cp140/full_weights.pt'
STRONG = WEAK.parent.parent / 'grpo_u_tau3_cp70/full_weights.pt'
FRACTIONS = {
    'first3': (.05, .95, .25, .75, .45, .55, .15, .85, .35, .65),
    'dense30': (.03, .30, .15, .06, .27, .12, .21, .09, .24, .18),
}


def freeze():
    """No outcome-based filtering; dedup all completed compatible weak-U states."""
    assert os.getuid() == 1003 and not BUNDLE.exists()
    pool = collect([OLD], chunk=20)
    assert pool['rows'] and all(r['state'][1][1] == str(WEAK) for r in pool['rows'])
    pools = {
        'dense30': deduplicate([pool]),
        'first3': deduplicate([{**pool, 'rows': [r for r in pool['rows'] if r['query'] < 3]}]),
    }
    for mode, p in pools.items():
        save(BUNDLE / (mode + '-calibration.json'), p)
    for batch in range(4):
        c = read(OLD / 'configs' / f'clean_weak_high_p25_b{batch}.json')
        save(BUNDLE / 'configs' / f'b{batch}.json', c)
        shutil.copy2(c['env']['seeds_path'], BUNDLE / 'configs' / f'seeds_b{batch}.json')
    for gpu in (4, 5):
        save(BUNDLE / f'environment-g{gpu}.json', read(OLD / f'environment-g{gpu}.json'))
    for name in ('clean_weak', 'tau3_strong'):
        for suffix in ('.json', '-episodes.json'):
            save(BUNDLE / 'summaries' / (name + suffix), read(OLD / 'summaries' / (name + suffix)))
    save(BUNDLE / 'frozen.json', dict(time=time.time(), source=str(OLD), outcome_blind=True,
         same_128_seeds_reused=True, exploratory=True,
         counts={m: dict(raw=p['raw_decisions'], unique=p['unique_decisions']) for m, p in pools.items()}))
    with tarfile.open(str(BUNDLE) + '.tar.gz', 'w:gz') as t:
        t.add(BUNDLE, arcname='.')
    print(json.dumps(read(BUNDLE / 'frozen.json')))


def prepare(mode):
    cards = (4, 5) if mode == 'first3' else (6, 7)
    host = 'sz1' if mode == 'first3' else 'sz3'
    assert os.getuid() == (1003 if host == 'sz1' else 20001)
    D = BASE / f'deployment-20261021/clean-c20-{mode}-{host}-v1'
    O = BASE / f'results/pi05-policy-eval/clean-c20-{mode}-20261021-v1'
    assert not (D / 'prepared.json').exists() and not O.exists()
    D.mkdir(parents=True, exist_ok=True)
    for directory in ('configs', 'thresholds', 'batches', 'summaries'):
        (O / directory).mkdir(parents=True)
    pool = read(BUNDLE / f'{mode}-calibration.json')
    rows = conditions(pool['rows'], fractions=FRACTIONS[mode])
    for row in rows:
        row['condition'] = row['condition'].replace('cp180_', mode + '_')
    save(O / 'calibration.json', pool)
    save(O / 'frozen-thresholds.json', dict(conditions=rows, frozen=read(BUNDLE / 'frozen.json'),
         unique_decisions=pool['unique_decisions'], max_takeover_queries=3 if mode == 'first3' else None))
    for p in (BUNDLE / 'summaries').glob('*.json'):
        value = read(p)
        if isinstance(value, dict):
            value.update(reused_from_host='sz1', reused_from=str(OLD / 'summaries' / p.name))
        save(O / 'summaries' / p.name, value)
    configs = []
    for lane, gpu in enumerate(cards):
        env = read(BUNDLE / f'environment-g{4+lane}.json')
        old_repo = env['REPO_PATH']
        env = {k: v.replace(old_repo, str(R)) if isinstance(v, str) else v for k,v in env.items()}
        env.update(REPO_PATH=str(R), CUDA_VISIBLE_DEVICES=str(gpu), DOJO_GPU_SCOPE=str(gpu),
                   TMPDIR=str(D / f'tmp-g{gpu}'))
        env.pop('RAY_ADDRESS', None)
        Path(env['TMPDIR']).mkdir(exist_ok=True)
        save(O / f'environment-g{gpu}.json', env)
        stages = []
        for row in rows:
            name = row['condition']; stage = dict(condition=name, configs=[])
            threshold = O / 'thresholds' / (name + '.json')
            save(threshold, {row['policy']: row['threshold'], **row})
            for batch in (lane, lane+2):
                c = copy.deepcopy(read(BUNDLE / 'configs' / f'b{batch}.json'))
                out = O / 'batches' / f'{name}_b{batch}'
                seeds = O / 'configs' / f'seeds_b{batch}.json'
                if not seeds.exists(): shutil.copy2(BUNDLE / 'configs' / seeds.name, seeds)
                c.update(condition=name, policy=row['policy'], gpu=gpu, output=str(out),
                         weak_weights=str(WEAK), strong_weights=str(STRONG),
                         threshold_path=str(threshold), sweep_target_fraction=row['target_fraction'],
                         max_takeover_queries=3 if mode == 'first3' else None,
                         comparison_baseline='clean_weak', collect_weak_u=True,
                         source_commit=(R / 'SOURCE_REVISION').read_text().strip())
                c['env']['seeds_path'] = str(seeds)
                c['env']['task_config']['save_path'] = str(out / 'env')
                c['env']['video_cfg']['video_base_dir'] = str(out / 'unused')
                validate_config(c)
                assert c['execute_chunk'] == 20 and c['num_steps'] == 10 and c['record_full_video']
                p = O / 'configs' / f'{name}_b{batch}.json'; save(p, c)
                configs.append(str(p)); stage['configs'].append(str(p))
            stages.append(stage)
        save(D / f'lane-g{gpu}-plan.json', dict(gpu=gpu, stages=stages))
    spec = dict(time=time.time(), source=str(R), output=str(O), mode=mode, cards=cards,
                weak=str(WEAK), strong=str(STRONG), H=50, C=20, N=10, step_limit=200,
                max_takeover_queries=3 if mode == 'first3' else None,
                per_card_parallelism=32, episodes_per_condition=128, conditions=rows,
                total_new_episodes=2560, unique_calibration_states=pool['unique_decisions'],
                configs=configs, reused_baselines=['clean_weak','tau3_strong'],
                target_is_calibration_fraction=True, plot_realized_fraction=True,
                stop='Inherited 600s stall / 3600s batch; original per-card RLT after exit and clearance')
    save(D / 'prepared.json', spec); save(O / 'manifest.json', spec)
    print(json.dumps({k:v for k,v in spec.items() if k not in ('configs','conditions')}))


if __name__ == '__main__':
    freeze() if sys.argv[1] == 'freeze' else prepare(sys.argv[1])
