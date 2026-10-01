"""Borrow SZ2 GPU 4--7 for EXPO, then return the exact four RLT runs.

Preparation is CPU-only. Stop/resume/status use the previous, already validated
rlt_cycle.py unchanged; this bridge only freezes the current resumed lineage.
Run with the original RLT Python, not the EXPO Python. No shared Ray restart.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import traceback

ROOT = Path('/data/chenyiteng')
KEYS = {'gpu4', 'gpu5', 'gpu6', 'gpu7'}
DEFAULT_PREVIOUS = ROOT / 'projects/robodojo-openwam/runs/rlt-cycle-20260930-single-v4'
WATCH = ROOT / 'deployment-20260927/rlt-six-task-watch/plan.json'
HELPER = 'rlt_cycle.py'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper(path):
    spec = importlib.util.spec_from_file_location('expo_frozen_rlt_cycle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def own_file(path, uid):
    path = Path(path).absolute()
    assert path.resolve().is_relative_to(ROOT.resolve()), 'File outside account root'
    assert not path.is_symlink() and path.is_file() and path.stat().st_uid == uid
    return path


def anchor(path):
    st = Path(path).stat()
    return dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def donor_checkpoints(m, previous, key, cfg, repo):
    """Only the same GPU's previously frozen, strictly valid whole checkpoints."""
    result = []; seen = set(); previous = Path(previous)
    for _ in range(16):
        assert str(previous) not in seen, 'Cycle lineage loop'
        seen.add(str(previous)); m.checked_dir(previous)
        plan = read(own_file(previous / 'plan.json', m.UID))
        stopped = read(own_file(previous / 'rlt-stopped.json', m.UID))
        assert plan['uid'] == m.UID and plan['head'] == m.EXPECTED_HEAD
        assert plan['repo'] == str(repo) and plan['runs'][key]['gpus'] == [int(key[3:])]
        cp = stopped['runs'][key]['checkpoint']; path = Path(cp['path'])
        for relative, digest in cp['small_sha256'].items():
            assert sha(own_file(path / 'actor' / relative, m.UID)) == digest, 'Frozen donor changed'
        checked = m.inspect_checkpoint(path, cfg, repo)
        assert checked['contract_sha256'] == cp['contract_sha256']
        replay = path / 'actor/sac_components/replay_buffer/rank_0'
        result.append(dict(path=path, checkpoint=checked, mapping=read(replay / 'trajectory_index.json')['trajectory_index']))
        prior = plan.get('predecessor_cycle')
        if not prior: break
        previous = Path(prior)
    else:
        raise RuntimeError('Unbounded donor lineage')
    return result


def sample_payloads(paths):
    import torch
    samples = []
    def tensors(value):
        if isinstance(value, torch.Tensor):
            assert value.numel() > 0
            if value.is_floating_point(): assert bool(torch.isfinite(value).all()), 'Nonfinite donor tensor'
            return 1
        if isinstance(value, dict): return sum(tensors(x) for x in value.values())
        if isinstance(value, (list, tuple)): return sum(tensors(x) for x in value)
        return 0
    for path in paths:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        assert isinstance(payload, dict) and tensors(payload) > 0, 'Bad trajectory payload'
        samples.append(dict(path=str(path), sha256=sha(path), dictionary_keys=sorted(payload)))
    return samples


def repair_latest(m, cp, cfg, repo, stage, previous, key):
    """Reconstruct missing old replay payloads in a new CP; never prune indices."""
    cp = Path(cp); replay = cp / 'actor/sac_components/replay_buffer/rank_0'
    metadata = read(own_file(replay / 'metadata.json', m.UID))
    index = read(own_file(replay / 'trajectory_index.json', m.UID))
    mapping = index['trajectory_index']; order = index['trajectory_id_list']
    assert len(order) == len(set(order)) == len(mapping) == metadata['size'] == metadata['total_samples']
    assert {str(x) for x in order} == set(mapping)
    expected = {}
    for tid, entry in mapping.items():
        assert entry['trajectory_id'] == int(tid) and entry['num_samples'] == 1
        model_id = entry['model_weights_id']
        assert isinstance(model_id, str) and re.fullmatch(r'[A-Za-z0-9-]+', model_id)
        expected['trajectory_' + tid + '_' + model_id + '.pt'] = tid
    present = {p.name: own_file(p, m.UID) for p in replay.glob('trajectory_*.pt')}
    assert set(present) <= set(expected), 'Repair supports missing-only, not extra/replaced payloads'
    assert all(p.stat().st_size > 0 for p in present.values()), 'Empty original payload'
    missing = sorted(set(expected) - set(present))
    assert missing, 'Failure is not a missing replay payload defect'
    target = ROOT / 'recovered-rlt' / stage.name / key / cp.name
    manifest_path = target / 'repair-manifest.json'
    original_small = {str(p.relative_to(cp / 'actor')): sha(p) for p in (
        cp / 'actor/sac_components/rlt_trainer_state/complete.json',
        cp / 'actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt',
        cp / 'actor/dcp_checkpoint/.metadata', replay / 'metadata.json', replay / 'trajectory_index.json')}
    if target.exists():
        manifest = read(own_file(manifest_path, m.UID))
        assert manifest['original_checkpoint'] == str(cp) and manifest['original_small_sha256'] == original_small
        assert manifest['complete'] is True
        return m.inspect_checkpoint(target, cfg, repo)
    donors = donor_checkpoints(m, previous, key, cfg, repo)
    sources = {}; rows = []
    for name in missing:
        tid = expected[name]
        for donor in donors:
            if donor['mapping'].get(tid) != mapping[tid]: continue
            candidate = donor['path'] / 'actor/sac_components/replay_buffer/rank_0' / name
            candidate = own_file(candidate, m.UID)
            assert candidate.stat().st_size > 0
            sources[name] = candidate
            rows.append(dict(name=name, source=str(candidate), source_anchor=anchor(candidate),
                             trajectory_entry_sha256=hashlib.sha256(json.dumps(mapping[tid], sort_keys=True).encode()).hexdigest()))
            break
    planning = dict(time=m.now(), key=key, original_checkpoint=str(cp), recovered_checkpoint=str(target),
                    indexed_samples=len(order), present_payloads=len(present), missing_payloads=len(missing),
                    covered_payloads=len(sources), full_coverage=len(sources) == len(missing),
                    original_small_sha256=original_small,
                    donors=[dict(path=str(d['path']), small_sha256=d['checkpoint']['small_sha256']) for d in donors],
                    payload_sources=rows)
    m.atomic(stage / ('repair-plan-' + key + '-' + cp.name + '.json'), planning)
    assert planning['full_coverage'], 'Missing payloads not fully covered by exact donor entries'
    # No directory or checkpoint copying until the read-only coverage plan is complete.
    payload_samples = sample_payloads([sources[missing[i]] for i in sorted({0, len(missing) // 2, len(missing) - 1})])
    assert target.resolve().is_relative_to(ROOT.resolve()) and not target.exists()
    target.mkdir(parents=True, mode=0o700)
    originals = []; inventory = hashlib.sha256(); linked = copied = 0
    for source in sorted(cp.rglob('*')):
        assert not source.is_symlink(), 'Symlink in original checkpoint'
        relative = source.relative_to(cp); dest = target / relative
        if source.is_dir(): dest.mkdir(exist_ok=True, mode=0o700); continue
        source = own_file(source, m.UID); before = anchor(source)
        originals.append((source, before))
        inventory.update(json.dumps(dict(relative=str(relative), anchor=before), sort_keys=True).encode() + b'\n')
        # All replay payloads and large immutable shards stay on this filesystem.
        if source.name.startswith('trajectory_') and source.suffix == '.pt' or before['bytes'] >= 1048576:
            os.link(source, dest); linked += 1
            assert anchor(dest) == before and anchor(source) == before
        else:
            shutil.copy2(source, dest); copied += 1
            assert sha(dest) == sha(source) and anchor(source) == before
    for row in rows:
        source = Path(row['source']); dest = target / 'actor/sac_components/replay_buffer/rank_0' / row['name']
        assert anchor(source) == row['source_anchor']
        os.link(source, dest)
        assert anchor(source) == anchor(dest) == row['source_anchor']
    for source, before in originals: assert anchor(source) == before, 'Original checkpoint mutated during clone'
    for relative, digest in original_small.items():
        assert sha(cp / 'actor' / relative) == sha(target / 'actor' / relative) == digest
    checked = m.inspect_checkpoint(target, cfg, repo)
    manifest = dict(**planning, complete=True, validated_at=m.now(), sampled_payloads=payload_samples,
                    original_file_anchor_sha256=inventory.hexdigest(), original_file_count=len(originals),
                    copied_small_files=copied, linked_original_files=linked, linked_missing_payloads=len(rows),
                    checked_step=checked['step'], checked_update_step=checked['update_step'],
                    contract_sha256=checked['contract_sha256'])
    m.save(manifest_path, manifest)
    return checked


def latest_valid(m, run, cfg, repo, audit, stage, previous, key):
    """A complete marker is necessary; all original strict checks must also pass."""
    candidates = sorted((p for p in m.checkpoint_root(run).glob('global_step_*')
                         if p.is_dir() and re.fullmatch(r'global_step_\d+', p.name)),
                        key=lambda p: int(p.name.rsplit('_', 1)[1]), reverse=True)
    records = audit['runs'].setdefault(str(run), [])
    for cp in candidates:
        marker = cp / 'actor/sac_components/rlt_trainer_state/complete.json'
        if not marker.is_file() or read(marker).get('complete') is not True:
            records.append(dict(path=str(cp), accepted=False, reason='No complete marker'))
            continue
        try:
            checked = m.inspect_checkpoint(cp, cfg, repo)
        except Exception as error:
            frame = traceback.extract_tb(error.__traceback__)[-1]
            records.append(dict(path=str(cp), accepted=False, error=type(error).__name__,
                                reason=str(error), check_line=frame.lineno, check=frame.line))
            m.atomic(Path(audit['path']), {k: v for k, v in audit.items() if k != 'path'})
            # Keep the latest model/optimizer/RNG/counters; do not fall back hundreds
            # of rounds when only older replay payloads were omitted by the writer.
            checked = repair_latest(m, cp, cfg, repo, stage, previous, key)
            records.append(dict(path=checked['path'], accepted=True, repaired_from=str(cp),
                                step=checked['step'], replay_samples=checked['replay_samples']))
            m.atomic(Path(audit['path']), {k: v for k, v in audit.items() if k != 'path'})
            return checked
        records.append(dict(path=str(cp), accepted=True, step=checked['step'],
                            replay_samples=checked['replay_samples'], contract_sha256=checked['contract_sha256']))
        m.atomic(Path(audit['path']), {k: v for k, v in audit.items() if k != 'path'})
        return checked
    m.atomic(Path(audit['path']), {k: v for k, v in audit.items() if k != 'path'})
    raise RuntimeError('No strictly valid complete checkpoint: ' + str(run))


def prepare(stage, previous):
    from omegaconf import OmegaConf
    previous = Path(previous).absolute()
    helper = own_file(previous / HELPER, 20001)
    m = load_helper(helper)
    m.checked_dir(previous); m.checked_dir(stage)
    assert not (stage / 'plan.json').exists(), 'Plan already exists; inspect it'
    assert not any(p.name != 'operation.lock' for p in stage.iterdir()), 'Nonempty new cycle'
    old = m.load_plan(previous)
    assert set(old['runs']) == KEYS and old['host'] == 'h100-gpu02'
    dispatched = read(own_file(previous / 'resumed-dispatched.json', m.UID))
    assert dispatched['cycle_id'] == previous.name
    assert {k: r['new_run'] for k, r in old['runs'].items()} == dispatched['runs']
    watch = read(own_file(WATCH, m.UID))
    live = m.actors(old)
    prepared = []
    audit = dict(path=str(stage / 'checkpoint-selection-prepare.json'), time=m.now(),
                 strict_validator_sha256=sha(helper), runs={})
    out = copy.deepcopy(old)
    out.update(cycle_id=stage.name, time=m.now(), script_sha256=sha(helper),
               management_namespace='expo-rlt-ops-' + stage.name[-40:],
               predecessor_cycle=str(previous), borrower='EXPO-FT', runs={})
    for key, row in old['runs'].items():
        gpu = int(key[3:])
        assert row['gpus'] == [gpu] and gpu in (4, 5, 6, 7)
        run = Path(row['new_run']); rt = run / 'runtime'
        matches = [r for r in watch['runs'].values() if r['run'] == str(run)]
        assert len(matches) == 1 and matches[0]['gpus'] == [gpu]
        assert matches[0]['namespace'] == row['namespace'], 'Current watch route changed'
        identity = read(own_file(rt / 'driver-identity.json', m.UID))
        launched = read(own_file(previous / (key + '-launched.json'), m.UID))['identity']
        assert m.same(identity) and identity['uid'] == m.UID
        assert (identity['pid'], identity['start']) == (launched['pid'], launched['start'])
        argv = (Path('/proc') / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0')
        assert str(helper).encode() in argv and b'driver' in argv
        assert argv.count(b'--cycle-dir') == argv.count(b'--key') == 1
        assert Path(os.fsdecode(argv[argv.index(b'--cycle-dir') + 1])).resolve() == previous.resolve()
        assert argv[argv.index(b'--key') + 1] == key.encode()
        identity.update(match_cmdline=True, cmdline_sha256=m.proc(identity['pid'])['cmdline_sha256'], boot=boot())
        selected = m.active(live, row['namespace']); jobs = {a['job_id'] for a in selected}
        assert selected and len(jobs) == 1
        m.validate_actor_rows(selected, row['namespace'], jobs)
        cfg = m.config(own_file(rt / 'resolved.yaml', m.UID))
        tb = run / 'tensorboard/config.yaml'
        if not tb.is_file(): tb = run / run.name / 'tensorboard/config.yaml'
        assert cfg == m.config(own_file(tb, m.UID)), 'Runtime differs from actual logged config'
        checked = latest_valid(m, run, cfg, old['repo'], audit, stage, previous, key)
        cp = Path(checked['path'])
        assert checked['gpu_used_for_check'] is False
        assert cfg['runner']['max_steps'] == cfg['runner']['max_epochs'] == 3000
        assert checked['step'] < 3000, 'Run already reached the accumulated target'
        target = ROOT / 'results/rlinf-rlt' / m.resumed_name(run, stage.name)
        namespace = 'er-' + stage.name[-40:] + '-g' + str(gpu)
        assert not target.exists() and not m.active(live, namespace)
        new_cfg, changes = m.resumed_config(cfg, run, target, cp)
        env = read(own_file(rt / 'environment.json', m.UID))
        assert not any(k in env for k in m.MASKS)
        env = {k: v.replace(str(run), str(target)).replace(row['namespace'], namespace) for k, v in env.items()}
        item = dict(task=row['task'], kind=row['kind'], gpus=[gpu], entry=row['entry'],
                    original_run=str(run), original_namespace=row['namespace'], original_identity=identity,
                    original_jobs=sorted(jobs), original_config_sha256=sha(rt / 'resolved.yaml'),
                    new_run=str(target), namespace=namespace, checkpoint=checked, config_changes=changes,
                    dependencies=m.dependency_snapshot(cfg), latest_metrics_before=m.latest_metrics(run))
        prepared.append((key, cfg, new_cfg, env, item))
    # No production process changes above. Freeze exactly the old helper bytes.
    shutil.copyfile(helper, stage / HELPER); (stage / HELPER).chmod(0o500)
    for key, cfg, new_cfg, env, item in prepared:
        pre = stage / 'prepared' / key; pre.mkdir(parents=True, mode=0o700)
        OmegaConf.save(OmegaConf.create(cfg), pre / 'original.yaml', resolve=True)
        OmegaConf.save(OmegaConf.create(new_cfg), pre / 'resolved.yaml', resolve=True)
        m.save(pre / 'environment.json', env)
        for p in pre.iterdir(): p.chmod(0o600)
        item['prepared_sha256'] = {n: sha(pre / n) for n in ('original.yaml', 'resolved.yaml', 'environment.json')}
        out['runs'][key] = item
    m.source_check(out)
    m.save(stage / 'plan.json', out)
    m.save(stage / 'bridge-frozen.json', dict(time=m.now(), boot=boot(), previous_cycle=str(previous),
        previous_plan_sha256=sha(previous / 'plan.json'), helper_sha256=sha(stage / HELPER),
        plan_sha256=sha(stage / 'plan.json'), bridge_sha256=sha(Path(__file__)),
        policy='Stop exact four RLT jobs; resume latest complete CP toward the original accumulated 3000 rounds.'))
    result = dict(prepared=True, cycle_id=stage.name, borrower='EXPO-FT', gpu_used_for_check=False,
        accumulated_target_rounds=3000, run_count=4,
        runs={k: dict(original_run=r['original_run'], original_namespace=r['original_namespace'],
                      checkpoint=r['checkpoint']['path'], checkpoint_step=r['checkpoint']['step'],
                      new_run=r['new_run'], namespace=r['namespace']) for k, r in out['runs'].items()})
    m.save(stage / 'prepared.json', result)
    return result


def frozen(stage):
    m = load_helper(own_file(stage / HELPER, 20001)); m.checked_dir(stage)
    receipt = read(own_file(stage / 'bridge-frozen.json', m.UID))
    assert receipt['boot'] == boot(), 'Host rebooted; inspect ownership before continuing'
    assert receipt['helper_sha256'] == sha(stage / HELPER)
    assert receipt['plan_sha256'] == sha(stage / 'plan.json')
    assert receipt['bridge_sha256'] == sha(Path(__file__)), 'EXPO bridge changed'
    m.load_plan(stage)
    return m


def stop_with_latest_valid(stage):
    """Scope selection to this imported module; keep frozen helper bytes intact."""
    m = frozen(stage); plan = m.load_plan(stage)
    if (stage / 'rlt-stopped.json').exists(): return m.stop(stage)
    audit = dict(path=str(stage / 'checkpoint-selection-stop.json'), time=m.now(),
                 strict_validator_sha256=sha(stage / HELPER), runs={})
    configs = {row['original_run']: (key, m.config(stage / 'prepared' / key / 'original.yaml'))
               for key, row in plan['runs'].items()}
    # Prove every run still has a usable whole checkpoint before any process stop.
    for run, (key, cfg) in configs.items():
        latest_valid(m, run, cfg, plan['repo'], audit, stage, plan['predecessor_cycle'], key)
    original = m.latest_complete
    def select(run):
        run = str(run)
        assert run in configs, 'Selection escaped the frozen four runs'
        key, cfg = configs[run]
        return Path(latest_valid(m, run, cfg, plan['repo'], audit, stage, plan['predecessor_cycle'], key)['path'])
    m.latest_complete = select
    try:
        return m.stop(stage)
    finally:
        m.latest_complete = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle-dir', type=Path, required=True)
    parser.add_argument('--previous-cycle', type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument('action', choices=('prepare', 'stop', 'resume', 'status'))
    parser.add_argument('--release-receipt', type=Path)
    args = parser.parse_args(); stage = args.cycle_dir.absolute()
    # The audited helper checks own UID, exact SZ2 hostname and path containment.
    check_path = args.previous_cycle / HELPER if args.action == 'prepare' else stage / HELPER
    check = load_helper(own_file(check_path, 20001)); check.checked_dir(stage)
    if args.action == 'status':
        result = frozen(stage).status(stage)
        if result['all_first_rounds_verified'] and not (stage / 'rlt-first-round.json').exists():
            check.save(stage / 'rlt-first-round.json', dict(state='verified', time=check.now(), status=result))
    else:
        import fcntl
        stage.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (stage / 'operation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'prepare': result = prepare(stage, args.previous_cycle)
            elif args.action == 'stop': result = stop_with_latest_valid(stage)
            else:
                assert args.release_receipt, 'EXPO owner must provide a terminal release receipt'
                result = frozen(stage).resume(stage, args.release_receipt)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
