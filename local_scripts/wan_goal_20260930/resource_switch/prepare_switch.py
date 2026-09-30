"""Prepare a fresh WM->Dojo owner after the old owner has returned four RLT jobs."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import time
from common import (HERE, HELPER, UID, atomic, bind_config, load_base,
                    load_module, name, own_path, read, sha)


def freeze_cycle(m, base, stage, previous):
    from omegaconf import OmegaConf
    previous = own_path(previous)
    old_module = load_module(previous / HELPER, 'previous_rlt_cycle')
    old = old_module.load_plan(previous)
    assert (previous / 'resumed-dispatched.json').is_file()
    m.checked_dir(stage); assert not stage.exists()
    out = copy.deepcopy(old)
    out.update(cycle_id=stage.name, time=m.now(), script_sha256=sha(base / HELPER),
               management_namespace='wm-dojo-rlt-' + stage.name[-40:],
               predecessor_cycle=str(previous), runs={})
    live = m.actors(old); prepared = []
    for key, row in old['runs'].items():
        run = Path(row['new_run']); rt = run / 'runtime'
        identity = read(rt / 'driver-identity.json')
        launch = read(previous / (key + '-launched.json'))['identity']
        assert m.same(identity) and identity['uid'] == UID
        assert (identity['pid'], identity['start']) == (launch['pid'], launch['start'])
        argv = (Path('/proc') / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0')
        assert str(previous / HELPER).encode() in argv and b'driver' in argv and key.encode() in argv
        identity.update(match_cmdline=True, cmdline_sha256=m.proc(identity['pid'])['cmdline_sha256'])
        actors = m.active(live, row['namespace']); jobs = {x['job_id'] for x in actors}
        assert len(jobs) == 1 and actors
        m.validate_actor_rows(actors, row['namespace'], jobs)
        cfg = m.config(rt / 'resolved.yaml'); recovery = m.select_recovery(run, cfg, old['repo'])
        assert recovery['mode'] == 'resume_checkpoint'
        cp = recovery['checkpoint']; target = m.ROOT / 'results/rlinf-rlt' / m.resumed_name(run, stage.name)
        namespace = 'wd-' + stage.name[-40:] + '-g' + key[3:]
        assert not target.exists() and not m.active(live, namespace)
        new_cfg, changes = m.resumed_config(cfg, run, target, cp['path'])
        env = read(rt / 'environment.json'); assert not any(k in env for k in m.MASKS)
        env = {k: v.replace(str(run), str(target)).replace(row['namespace'], namespace) for k, v in env.items()}
        item = dict(task=row['task'], kind=row['kind'], gpus=row['gpus'], entry=row['entry'],
                    original_run=str(run), original_namespace=row['namespace'], original_identity=identity,
                    original_jobs=sorted(jobs), original_config_sha256=sha(rt / 'resolved.yaml'),
                    new_run=str(target), namespace=namespace, config_changes=changes, recovery=recovery,
                    dependencies=m.dependency_snapshot(cfg), latest_metrics_before=m.latest_metrics(run))
        prepared.append((key, cfg, new_cfg, env, item))
    assert {x[0] for x in prepared} == {'gpu4', 'gpu5', 'gpu6', 'gpu7'}
    stage.mkdir(mode=0o700)
    shutil.copyfile(base / HELPER, stage / HELPER); (stage / HELPER).chmod(0o500)
    for key, cfg, new_cfg, env, item in prepared:
        pre = stage / 'prepared' / key; pre.mkdir(parents=True, mode=0o700)
        OmegaConf.save(OmegaConf.create(cfg), pre / 'original.yaml', resolve=True)
        OmegaConf.save(OmegaConf.create(new_cfg), pre / 'resolved.yaml', resolve=True)
        m.save(pre / 'environment.json', env)
        for file in pre.iterdir(): file.chmod(0o600)
        item['prepared_sha256'] = {n: sha(pre / n) for n in ('original.yaml', 'resolved.yaml', 'environment.json')}
        out['runs'][key] = item
    m.save(stage / 'plan.json', out)
    m.save(stage / 'prepared.json', dict(time=m.now(), cycle=str(stage), predecessor=str(previous)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base-source-dir', type=Path, required=True)
    parser.add_argument('--cycle-dir', type=Path, required=True)
    parser.add_argument('--continuation-name', required=True)
    parser.add_argument('--wm-spec', type=Path, required=True)
    args = parser.parse_args()
    base, m = load_base(args.base_source_dir)
    from dojo_sweep import Sweep, build_plan
    from resume_results import prepare_result
    from wm_stage import validate_spec
    config, cfg, project, run = bind_config(args.config)
    stage = own_path(args.cycle_dir, exists=False)
    assert stage.is_relative_to(project) and not stage.exists()
    attempt = run / name(args.continuation_name)
    prep = run / ('prepare-' + args.continuation_name)
    assert not attempt.exists() and not prep.exists()
    spec = validate_spec(read(own_path(args.wm_spec)))
    prior = read(run / 'plan.json'); plan = build_plan(cfg)
    assert cfg == prior['config'], 'Dojo configuration must stay identical'
    assert plan['episode_total'] == 6300 and plan['gpus'] == [4, 5, 6, 7] and plan['num_envs'] == 4
    for key in set(prior) | set(plan):
        if key != 'created_at': assert prior.get(key) == plan.get(key), 'Changed benchmark field: ' + key
    with (run / 'pipeline.lock').open('a+') as outer, (run / 'controller.lock').open('a+') as inner:
        fcntl.flock(outer, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(inner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = read(run / 'active-continuation.json')
        previous = Path(active['cycle_dir']); previous_attempt = Path(active['attempt_dir'])
        assert read(previous_attempt / 'pipeline-final.json')['rlt_dispatched'] is True
        first = read(previous_attempt / 'rlt-first-round.json')
        assert first['state'] == 'verified' and first['status']['all_first_rounds_verified'] is True
        assert not Sweep(cfg, prior).guard.scan(), 'Dojo processes still alive'
        prep.mkdir(mode=0o700)
        for file in ('plan.json', 'active-continuation.json', 'pipeline-current.json', 'result-audit.json'):
            if (run / file).exists(): shutil.copy2(run / file, prep / file)
        rows = [prepare_result(Path(cfg['repo']), task, seed,
                    f"{cfg['run_id']}_s{seed}_{task}", prior['budgets'][task], prep / 'preserved-results', apply=True)
                for seed in prior['seeds'] for task in prior['tasks']]
        atomic(prep / 'preserved-results.json', rows)
        freeze_cycle(m, base, stage, previous)
        atomic(prep / 'wm-spec.json', spec)
        ready = dict(time=time.time(), config_path=str(config), config_sha256=sha(config),
                     base_source_dir=str(base), cycle_dir=str(stage), attempt_dir=str(attempt),
                     prior_plan_sha256=sha(run / 'plan.json'), previous_cycle=str(previous),
                     wm_spec_sha256=sha(prep / 'wm-spec.json'), benchmark_unchanged=True,
                     source_sha256={p.name: sha(p) for p in HERE.iterdir() if p.suffix in ('.py', '.json')},
                     preserved_episodes=sum(x.get('episodes', 0) for x in rows))
        atomic(prep / 'ready.json', ready)
    print(json.dumps(dict(preparation=str(prep), **ready)))


if __name__ == '__main__':
    main()
