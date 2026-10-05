"""One SZ2 Dojo borrowing cycle: freeze, stop exact RLT jobs, then resume.

No shared Ray lifecycle changes; no broad process-name matching. The caller must
run this with the original RLT Python. See RLT_CYCLE.md for the release receipt.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import runpy
import shlex
import signal
import socket
import subprocess
import sys
import time
import types
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path('/data/chenyiteng')
TASKS = ('place_object_stand', 'move_playingcard_away')
EXPECTED_HEAD = 'fcd22b856d72d810c663417bef20610679e2358f'
UID = 20001
MASKS = ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')
CODE_FILES = ('rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py',
              'rlinf/workers/actor/fsdp_sac_policy_worker.py',
              'rlinf/hybrid_engines/fsdp/strategy/checkpoint.py',
              'rlinf/runners/embodied_runner.py',
              'rlinf/data/storage/replay/buffer.py')


def now():
    return datetime.datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, payload, mode=0o600):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, 'w') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def atomic(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + '.dojo-rlt-tmp-' + str(os.getpid()))
    save(temp, payload, path.stat().st_mode & 0o777 if path.exists() else 0o600)
    os.replace(temp, path)


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def checked_dir(path):
    path = Path(path).absolute()
    assert re.fullmatch(r'[a-zA-Z0-9_-]{6,100}', path.name), 'Unsafe cycle name'
    resolved = path.resolve()
    assert resolved.is_relative_to(ROOT.resolve()), 'Cycle escapes own data directory'
    assert path == resolved or path.resolve() == resolved
    if path.exists():
        assert not path.is_symlink() and path.stat().st_uid == UID
    assert os.getuid() == UID and socket.gethostname() == 'h100-gpu02', 'Wrong host/account'
    return path


def proc(pid):
    try:
        path = Path('/proc') / str(int(pid))
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'uid': path.stat().st_uid, 'start': int(fields[19]),
                'state': fields[0], 'ppid': int(fields[1]),
                'cmdline_sha256': hashlib.sha256((path/'cmdline').read_bytes()).hexdigest()}
    except (OSError, ValueError):
        return None


def same(identity):
    current = proc(identity['pid'])
    start = identity.get('start', identity.get('start_ticks'))
    return bool(current and current['state'] not in ('Z', 'X') and
                current['uid'] == identity['uid'] and current['start'] == start and
                (not identity.get('match_cmdline') or current.get('cmdline_sha256') == identity['cmdline_sha256']))


def gpu_processes(gpus):
    tree = ET.fromstring(command(['nvidia-smi', '-q', '-x']))
    wanted = set(gpus)
    result = []
    for index, gpu in enumerate(tree.findall('gpu')):
        if index not in wanted:
            continue
        for entry in gpu.findall('./processes/process_info'):
            pid = entry.findtext('pid', '')
            if pid.isdigit():
                result.append({'gpu': index, 'uuid': gpu.findtext('uuid'),
                               'pid': int(pid), 'type': entry.findtext('type')})
    return result


def actors(plan):
    url = plan['ray_dashboard_url'].rstrip('/') + '/api/v0/actors?limit=10000&detail=1'
    with urllib.request.urlopen(url, timeout=25) as response:
        payload = json.load(response)
    result = payload['data']['result']
    assert result.get('num_after_truncation', result.get('total', 0)) < 10000
    return result['result']


def active(rows, namespace):
    return [r for r in rows if r.get('ray_namespace') == namespace and r.get('state') != 'DEAD']


def process_tree(roots):
    table = {int(p.name): proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    table = {k: v for k, v in table.items() if v}
    selected = set(roots) & table.keys()
    for _ in range(64):
        added = {pid for pid, row in table.items() if row['ppid'] in selected}
        if added <= selected:
            break
        selected |= added
    else:
        raise RuntimeError('Unbounded process ancestry')
    assert all(table[pid]['uid'] == UID for pid in selected), 'Foreign UID in target tree'
    return {pid: table[pid] for pid in selected}


def source_check(plan):
    repo = Path(plan['repo'])
    assert command(['git', '-C', str(repo), 'rev-parse', 'HEAD']) == plan['head'] == EXPECTED_HEAD
    assert not command(['git', '-C', str(repo), 'diff', '--name-only', 'HEAD', '--', 'rlinf', 'examples']), 'Production source changed'
    for relative, expected in plan['source_sha256'].items():
        assert sha(repo / relative) == expected, relative


def config(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def runtime_contract(repo, cfg):
    """Execute only the audited pure contract function, without constructing a worker."""
    from omegaconf import OmegaConf
    path = Path(repo) / CODE_FILES[0]
    tree = ast.parse(path.read_text())
    names = ('_rlt_contract', '_rlt_dvac_success_scale_schedule')
    selected = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == 2 and all(not n.decorator_list for n in selected)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    scope = {'OmegaConf': OmegaConf, 'json': json, 'hashlib': hashlib, 'math': math}
    exec(compile(module, str(path), 'exec'), scope)
    dvac = cfg['algorithm'].get('rlt_dvac', {}) or {}
    shim = types.SimpleNamespace(rlt_resume_cfg=copy.deepcopy(cfg['algorithm']['rlt_resume']),
        rlt_dvac_cfg=copy.deepcopy(dvac), rlt_dvac_mode=dvac.get('mode', 'off'), cfg=OmegaConf.create(cfg))
    shim._rlt_dvac_success_scale_schedule = types.MethodType(scope['_rlt_dvac_success_scale_schedule'],shim)
    return scope['_rlt_contract'](shim)


def checkpoint_root(run):
    run = Path(run)
    candidates = [run / run.name / 'checkpoints', run / 'checkpoints']
    existing = [p for p in candidates if p.is_dir()]
    assert len(existing) == 1, 'Ambiguous checkpoint root'
    return existing[0]


def latest_complete(run):
    candidates = sorted((p for p in checkpoint_root(run).glob('global_step_*')
        if p.is_dir() and re.fullmatch(r'global_step_\d+', p.name)),
        key=lambda p: int(p.name.rsplit('_', 1)[1]), reverse=True)
    for cp in candidates:
        marker = cp / 'actor/sac_components/rlt_trainer_state/complete.json'
        if marker.is_file() and read(marker).get('complete') is True:
            return cp
    raise RuntimeError('No complete checkpoint: ' + str(run))


def inspect_checkpoint(cp, cfg, repo):
    """CPU-only strict small-state and DCP/index validation; no GPU allocation."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    cp = Path(cp)
    actor = cp / 'actor'
    state_dir = actor / 'sac_components/rlt_trainer_state'
    marker = read(state_dir / 'complete.json')
    serialized, digest = runtime_contract(repo, cfg)
    step = int(cp.name.rsplit('_', 1)[1])
    assert cfg['algorithm']['rlt_resume']['enable'] is True
    assert cfg['actor']['fsdp_config']['use_orig_params'] is False, 'Expected verified DCP format'
    assert marker['complete'] is True and marker['schema_version'] == 1
    assert marker['actor_world_size'] == 1 and marker['saved_runner_step'] == step
    assert marker['rank_files'] == ['checkpoint_rank_0.pt']
    assert marker['rlt_resume_contract_sha256'] == digest, 'Effective resume contract mismatch'
    state = torch.load(state_dir / 'checkpoint_rank_0.pt', map_location='cpu', weights_only=True)
    assert state['rank'] == 0 and state['actor_world_size'] == 1
    for key in ('schema_version', 'saved_runner_step', 'update_step', 'rlt_resume_contract_sha256'):
        assert state[key] == marker[key], key
    assert state['rlt_resume_contract'] == serialized
    for key in ('update_step', 'local_total_transitions_added', 'local_total_episodes_added'):
        assert isinstance(state[key], int) and state[key] >= 0, key
    for key in ('global_warmup_ready_total_transitions', 'global_warmup_ready_total_episodes'):
        assert state[key] is not None and state[key] >= 0, 'Expected mature online RLT checkpoint'
    dcp_dir = actor / 'dcp_checkpoint'
    dcp = FileSystemReader(dcp_dir).read_metadata()
    entries = list(dcp.state_dict_metadata)
    for component in ('model', 'optimizers', 'lr_schedulers', 'rng'):
        assert any(component in name.split('.') for name in entries), 'DCP lacks ' + component
    shards = {}
    for item in dcp.storage_data.values():
        path = dcp_dir / item.relative_path
        assert path.resolve().is_relative_to(dcp_dir.resolve())
        size = path.stat().st_size
        assert item.offset >= 0 and item.length > 0 and item.offset + item.length <= size
        shards[item.relative_path] = size
    target = actor / 'sac_components/target_model/checkpoint_rank_0.pt'
    assert target.is_file() and target.stat().st_size > 0
    replay = actor / 'sac_components/replay_buffer/rank_0'
    metadata = read(replay / 'metadata.json')
    index = read(replay / 'trajectory_index.json')
    assert set(index) == {'trajectory_index', 'trajectory_id_list'}
    mapping = index['trajectory_index']; order = index['trajectory_id_list']
    assert isinstance(mapping, dict) and isinstance(order, list)
    assert len(order) == len(set(order)) == len(mapping) == metadata['size'] == metadata['total_samples']
    files = {p.name: p.stat().st_size for p in replay.glob('trajectory_*.pt') if p.is_file()}
    assert len(files) == len(order) and all(n > 0 for n in files.values())
    assert {str(n) for n in order} == set(mapping)
    for key, entry in mapping.items():
        assert isinstance(entry, dict), 'Unexpected replay index schema'
        assert entry['trajectory_id'] == int(key) and entry['num_samples'] == 1
        model_id = entry['model_weights_id']
        assert isinstance(model_id,str) and re.fullmatch(r'[A-Za-z0-9-]+',model_id)
        sample = replay / ('trajectory_'+key+'_'+model_id+'.pt')
        assert sample.name in files and sample.is_file(), 'Missing indexed replay sample'
    small_hashes = {str(p.relative_to(actor)): sha(p) for p in
        (state_dir/'complete.json', state_dir/'checkpoint_rank_0.pt',
         dcp_dir/'.metadata', replay/'metadata.json', replay/'trajectory_index.json')}
    return {'path': str(cp), 'step': step, 'update_step': state['update_step'],
        'replay_samples': len(order), 'replay_bytes': sum(files.values()),
        'contract_sha256': digest, 'small_sha256': small_hashes, 'dcp_shards': shards,
        'target_bytes': target.stat().st_size, 'validated_at': now(), 'gpu_used_for_check': False}


def diff(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        for key in set(a) | set(b):
            name = (prefix + '.' + str(key)).strip('.')
            result.update(diff(a.get(key), b.get(key), name))
        return result
    return {} if a == b else {prefix: [a, b]}


def resumed_config(original, old_run, new_run, cp):
    value = copy.deepcopy(original)
    def replace(v):
        if isinstance(v, dict): return {k: replace(x) for k, x in v.items()}
        if isinstance(v, list): return [replace(x) for x in v]
        return v.replace(str(old_run), str(new_run)) if isinstance(v, str) else v
    value = replace(value)
    value['runner']['logger']['experiment_name'] = Path(new_run).name
    value['runner']['resume_dir'] = str(cp)
    changes = diff(original, value)
    allowed = {'runner.logger.log_path', 'runner.logger.experiment_name', 'runner.resume_dir'} | {
        f'env.{phase}.{field}' for phase in ('train','eval') for field in
        ('task_config.save_path','video_cfg.video_base_dir')}
    assert set(changes) <= allowed, 'Unexpected config changes: ' + str(sorted(set(changes)-allowed))
    assert value['algorithm'] == original['algorithm']
    assert value['runner']['max_steps'] == value['runner']['max_epochs'] == 3000
    return value, changes


def latest_metrics(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run = Path(run)
    tb = run / 'tensorboard'
    if not tb.is_dir(): tb = run / run.name / 'tensorboard'
    if not tb.is_dir(): return {}
    ea = EventAccumulator(str(tb), size_guidance={'scalars': 10}); ea.Reload()
    output = {}
    for tag in ea.Tags()['scalars']:
        if tag == 'env/success_once' or any(k in tag for k in
            ('ready_for_online','update_step','global_min_replay_size','critic_updates_run')):
            values = ea.Scalars(tag)
            if values:
                x = values[-1]; output[tag] = {'round': x.step + 1, 'value': x.value}
    return output


def dependency_snapshot(cfg):
    feature=Path(cfg['rollout']['rlt_feature_model']['model_path'])
    candidates=[feature/'actor/model_state_dict/full_weights.pt',
                feature/'model_state_dict/full_weights.pt',feature]
    weights=[p for p in candidates if p.is_file()]
    assert len(weights)==1 and weights[0].stat().st_size>0,'Missing or ambiguous Stage1 initialization weights'
    st=weights[0].stat()
    output={'stage1':{'path':str(weights[0]),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns},'small_sha256':{}}
    for phase in ('train','eval'):
        assert Path(cfg['env'][phase]['assets_path']).is_dir()
        seed=Path(cfg['env'][phase]['seeds_path'])
        output['small_sha256'][str(seed)]=sha(seed)
    norm=Path(cfg['rollout']['rlt_feature_model']['openpi_data']['norm_stats_path'])
    output['small_sha256'][str(norm)]=sha(norm)
    return output


def load_plan(stage):
    plan = read(stage / 'plan.json')
    assert plan['cycle_id'] == stage.name and plan['uid'] == UID
    assert sha(Path(__file__)) == plan['script_sha256'], 'Cycle script changed'
    source_check(plan)
    return plan


def resumed_name(old_run, cycle):
    old=Path(old_run).name
    original=old.split('-after-dojo-',1)[0]
    # Hash the full lineage to distinguish retries without appending it forever.
    digest=hashlib.sha256((old+'\0'+str(cycle)).encode()).hexdigest()[:12]
    label=original.encode()[:100].decode(errors='ignore')
    cycle_label=Path(cycle).name.encode()[:50].decode(errors='ignore')
    result=label+'-after-dojo-'+cycle_label+'-'+digest
    assert len(result.encode())<=180 and '/' not in result
    return result


def prepare(stage):
    from omegaconf import OmegaConf
    assert not (stage/'plan.json').exists(), 'Preparation already exists; inspect it'
    stage.mkdir(parents=True, exist_ok=True, mode=0o700)
    plans = [read(ROOT/'deployment-20261002'/('rlt-next6-'+task)/'plan.json') for task in TASKS]
    assert len({p['repo'] for p in plans}) == len({p['python'] for p in plans}) == 1
    original = plans[0]; repo = Path(original['repo'])
    result = {'cycle_id': stage.name, 'time': now(), 'uid': UID, 'host': socket.gethostname(),
        'repo': str(repo), 'head': EXPECTED_HEAD, 'python': original['python'],
        'ray_address': original['ray_address'], 'ray_dashboard_url': original['ray_dashboard_url'],
        'script_sha256': sha(Path(__file__)), 'source_sha256': {p: sha(repo/p) for p in CODE_FILES},
        'management_namespace': 'dojo-rlt-ops-' + stage.name[-40:], 'runs': {}}
    source_check(result)
    frozen = stage/'rlt_cycle.py'
    if frozen.resolve() != Path(__file__).resolve():
        assert not frozen.exists()
        frozen.write_bytes(Path(__file__).read_bytes()); frozen.chmod(0o500)
    live_actors = actors(result)
    for p in plans:
        assert p['head'] == EXPECTED_HEAD and p['uid'] == UID
        assert p['ray_address'] == result['ray_address'] and p['ray_dashboard_url'] == result['ray_dashboard_url']
        for kind in ('clean','combo'):
            row = p['runs'][kind]; old_run = Path(row['run']); rt = old_run/'runtime'
            identity = read(rt/'driver-identity.json')
            assert same(identity) and identity['uid'] == UID, 'Original driver no longer matches'
            argv=(Path('/proc')/str(identity['pid'])/'cmdline').read_bytes().split(b'\0')
            assert str(p['ops']).encode() in argv and kind.encode() in argv and b'driver' in argv
            original_stage=ROOT/'deployment-20261002'/('rlt-next6-'+p['task'])
            assert argv.count(b'--stage') == 1, 'Original driver has ambiguous stage argument'
            actual_stage=Path(os.fsdecode(argv[argv.index(b'--stage')+1]))
            assert actual_stage.resolve() == original_stage.resolve(), 'Original driver command belongs to another run'
            identity.update(cmdline_sha256=proc(identity['pid'])['cmdline_sha256'],match_cmdline=True)
            assert len(row['gpus']) == 1 and row['gpus'][0] in (4,5,6,7)
            owned = active(live_actors, row['namespace'])
            jobs = {a['job_id'] for a in owned}
            assert len(jobs) == 1 and owned
            cfg = config(rt/'resolved.yaml')
            tb = old_run/'tensorboard/config.yaml'
            if not tb.is_file(): tb = old_run/old_run.name/'tensorboard/config.yaml'
            assert cfg == config(tb), 'Actual config differs from frozen runtime'
            cp = latest_complete(old_run)
            checked = inspect_checkpoint(cp, cfg, repo)
            gpu = row['gpus'][0]; key = 'gpu' + str(gpu)
            new_run = ROOT/'results/rlinf-rlt'/(resumed_name(old_run,stage.name))
            namespace = 'dr-'+stage.name[-40:]+'-g'+str(gpu)
            assert not new_run.exists() and not active(live_actors, namespace)
            cfg_new, changes = resumed_config(cfg, old_run, new_run, cp)
            pre = stage/'prepared'/key; pre.mkdir(parents=True, mode=0o700)
            OmegaConf.save(OmegaConf.create(cfg), pre/'original.yaml', resolve=True)
            OmegaConf.save(OmegaConf.create(cfg_new), pre/'resolved.yaml', resolve=True)
            # Private runtime transfer, never logged or published.
            environment = read(rt/'environment.json')
            assert not any(k in environment for k in MASKS)
            environment = {k: v.replace(str(old_run),str(new_run)).replace(row['namespace'],namespace)
                           for k,v in environment.items()}
            save(pre/'environment.json', environment)
            result['runs'][key] = {'task': p['task'], 'kind': kind, 'gpus': [gpu],
                'original_run': str(old_run), 'original_namespace': row['namespace'],
                'original_identity': identity, 'original_jobs': sorted(jobs),
                'original_config_sha256': sha(rt/'resolved.yaml'),
                'new_run': str(new_run), 'namespace': namespace, 'entry': row['entry'],
                'checkpoint': checked, 'config_changes': changes,
                'dependencies':dependency_snapshot(cfg),
                'latest_metrics_before': latest_metrics(old_run),
                'prepared_sha256': {n:sha(pre/n) for n in ('original.yaml','resolved.yaml','environment.json')}}
    assert set(result['runs']) == {'gpu4','gpu5','gpu6','gpu7'}
    save(stage/'plan.json', result)
    save(stage/'prepared.json', {'time':now(),'runs':{k:{'checkpoint':r['checkpoint']['path'],
        'new_run':r['new_run'],'namespace':r['namespace']} for k,r in result['runs'].items()}})
    return {'prepared':True,'cycle_id':stage.name,'run_count':4}


def validate_actor_rows(rows, namespace, jobs):
    for actor in rows:
        assert actor['ray_namespace'] == namespace and actor['job_id'] in jobs
        assert actor.get('name') and actor.get('actor_id'), 'Unaddressable actor'
        identity = proc(actor.get('pid',0))
        if identity and identity['state'] not in ('Z','X'):
            assert identity['uid'] == UID, 'Foreign actor UID'


def kill_actors(plan, rows):
    import ray
    assert not ray.is_initialized()
    ray.init(address=plan['ray_address'],namespace=plan['management_namespace'],log_to_driver=False)
    try:
        for actor in rows:
            try:
                handle=ray.get_actor(actor['name'],namespace=actor['ray_namespace'])
            except ValueError:
                continue
            assert handle._actor_id.hex() == actor['actor_id'], 'Actor identity changed'
            ray.kill(handle,no_restart=True)
    finally:
        ray.shutdown()


def stop(stage):
    from omegaconf import OmegaConf
    plan=load_plan(stage)
    if (stage/'rlt-stopped.json').exists():
        return {'already_stopped':True,'receipt':read(stage/'rlt-stopped.json')}
    assert not (stage/'stop-attempt.json').exists(), 'Prior stop needs inspection; no automatic replay'
    live=actors(plan); identities={}; snapshots={}
    for key,row in plan['runs'].items():
        assert same(row['original_identity']), 'Original driver identity changed'
        assert sha(Path(row['original_run'])/'runtime/resolved.yaml') == row['original_config_sha256']
        selected=active(live,row['original_namespace'])
        validate_actor_rows(selected,row['original_namespace'],set(row['original_jobs']))
        roots={row['original_identity']['pid']} | {a['pid'] for a in selected if a.get('pid')}
        tree=process_tree(roots)
        assert {r['pid'] for r in gpu_processes(row['gpus'])} <= set(tree), 'Unrelated GPU process'
        identities.update(tree)
        snapshots[key]={'actors':selected,'driver':row['original_identity'],'jobs':row['original_jobs']}
    save(stage/'stop-attempt.json',{'time':now(),'runs':snapshots,'processes':list(identities.values()),
         'policy':'User authorized immediate stop and resume from latest complete CP; no wait for next save.'})
    for row in plan['runs'].values():
        if same(row['original_identity']):os.kill(row['original_identity']['pid'],signal.SIGTERM)
    time.sleep(10)
    remaining=[]; live=actors(plan)
    for row in plan['runs'].values():
        selected=active(live,row['original_namespace'])
        validate_actor_rows(selected,row['original_namespace'],set(row['original_jobs']))
        remaining.extend(selected)
        identities.update(process_tree({a['pid'] for a in selected if a.get('pid')}))
    if remaining:
        save(stage/'scoped-actor-cleanup.json',{'time':now(),'actors':remaining})
        kill_actors(plan,remaining)
    for _ in range(10):
        if not any(same(v) for v in identities.values()):break
        time.sleep(1)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        identities.update(process_tree({pid for pid,v in identities.items() if same(v)}))
        targets=[v for v in identities.values() if same(v)]
        if not targets:break
        scoped={r['original_namespace']:set(r['original_jobs']) for r in plan['runs'].values()}
        target_pids={v['pid'] for v in targets}
        for actor in actors(plan):
            if actor.get('state')!='DEAD' and actor.get('pid') in target_pids:
                assert actor.get('ray_namespace') in scoped and actor.get('job_id') in scoped[actor['ray_namespace']]
        save(stage/('signal-'+str(int(sig))+'.json'),{'time':now(),'processes':targets})
        for v in targets:
            if same(v):os.kill(v['pid'],sig)
        for _ in range(10):
            if not any(same(v) for v in targets):break
            time.sleep(1)
    for _ in range(60):
        if not gpu_processes([4,5,6,7]):break
        time.sleep(1)
    assert not gpu_processes([4,5,6,7]), 'GPU contexts still active; Dojo must not start'
    live=actors(plan)
    assert not any(active(live,r['original_namespace']) for r in plan['runs'].values())
    assert not any(same(r['original_identity']) for r in plan['runs'].values())
    frozen={}
    for key,row in plan['runs'].items():
        pre=stage/'prepared'/key;cfg=config(pre/'original.yaml')
        cp=latest_complete(row['original_run']);checked=inspect_checkpoint(cp,cfg,plan['repo'])
        cfg_new,changes=resumed_config(cfg,row['original_run'],row['new_run'],cp)
        OmegaConf.save(OmegaConf.create(cfg_new),pre/'resolved.yaml',resolve=True)
        metrics=latest_metrics(row['original_run'])
        collection=metrics.get('env/success_once',{}).get('round')
        frozen[key]={'checkpoint':checked,'resolved_sha256':sha(pre/'resolved.yaml'),
                     'config_changes':changes,'last_metrics':metrics,
                     'collection_rounds_after_checkpoint':None if collection is None else collection-checked['step']}
    receipt={'time':now(),'cycle_id':stage.name,'runs':frozen,'all_original_drivers_stopped':True,
             'all_original_namespaces_empty':True,'gpus_released':[4,5,6,7]}
    save(stage/'rlt-stopped.json',receipt)
    return {'stopped':True,'checkpoints':{k:v['checkpoint']['step'] for k,v in frozen.items()}}


def guard_names(plan, stage):
    path=ROOT/'security/ray-guard/training_allowlist.json'
    assert not path.is_symlink() and path.stat().st_uid == UID
    before=read(path); names={r['namespace'] for r in plan['runs'].values()}
    if names <= set(before['namespaces']):return
    if not (stage/'allowlist-before.json').exists():save(stage/'allowlist-before.json',before)
    after=dict(before);after['namespaces']=sorted(set(before['namespaces'])|names)
    atomic(path,after)
    save(stage/'allowlist-added.json',{'time':now(),'added':sorted(names-set(before['namespaces']))})


def update_watch(plan,stage):
    path=ROOT/'deployment-20260927/rlt-six-task-watch/plan.json'
    assert not path.is_symlink() and path.stat().st_uid==UID
    before=read(path);after=copy.deepcopy(before)
    found=set()
    for key,target in after['runs'].items():
        for run_key,row in plan['runs'].items():
            if target['run'] in (row['original_run'],row['new_run']):
                assert target['gpus']==row['gpus']
                target.update(run=row['new_run'],namespace=row['namespace'])
                found.add(run_key)
    assert found==set(plan['runs']), 'Watch route no longer matches frozen four runs'
    if before==after:return
    if not (stage/'watch-before.json').exists():save(stage/'watch-before.json',before)
    atomic(path,after)
    save(stage/'watch-updated.json',{'time':now(),'path':str(path),'tracked_resume_runs':sorted(found)})


def validate_release(stage, receipt_path):
    receipt_path=Path(receipt_path)
    assert receipt_path.resolve().is_relative_to(ROOT.resolve()) and receipt_path.stat().st_uid==UID
    receipt=read(receipt_path)
    assert receipt['cycle_id']==stage.name
    assert receipt['terminal_status'] in ('completed','failed','timed_out','not_started')
    assert receipt['all_workers_stopped'] is True
    managed=receipt['managed_processes']
    assert isinstance(managed,list) and (managed or receipt['terminal_status']=='not_started')
    for identity in managed:
        assert identity['uid']==UID and not same(identity), 'Dojo process still alive'
    assert not gpu_processes([4,5,6,7]), 'GPU not released after Dojo'
    return {'path':str(receipt_path),'sha256':sha(receipt_path),'terminal_status':receipt['terminal_status']}


def resume(stage,release_receipt):
    plan=load_plan(stage);stopped=read(stage/'rlt-stopped.json')
    if (stage/'resumed-dispatched.json').exists():return status(stage)
    if (stage/'dojo-release-verified.json').exists():
        release=read(stage/'dojo-release-verified.json')
        assert str(Path(release_receipt))==release['path'] and sha(release_receipt)==release['sha256']
    else:
        release=validate_release(stage,release_receipt)
        save(stage/'dojo-release-verified.json',release)
    live=actors(plan)
    assert not any(same(r['original_identity']) or active(live,r['original_namespace']) for r in plan['runs'].values())
    guard_names(plan,stage)
    for key,row in plan['runs'].items():
        rt=Path(row['new_run'])/'runtime';pre=stage/'prepared'/key
        if (stage/(key+'-launched.json')).exists():
            launched=read(stage/(key+'-launched.json'))
            assert same(launched['identity']), 'Previously launched resume driver is not alive'
            continue
        attempt=stage/(key+'-launch-attempt.json')
        if attempt.exists():
            # Fail closed after ambiguous dispatch; do not duplicate a Ray namespace.
            raise RuntimeError('Ambiguous previous dispatch for '+key+'; inspect identity and namespace')
        assert not active(actors(plan),row['namespace']) and not gpu_processes(row['gpus'])
        assert sha(pre/'resolved.yaml')==stopped['runs'][key]['resolved_sha256']
        assert sha(pre/'environment.json')==row['prepared_sha256']['environment.json']
        cp=stopped['runs'][key]['checkpoint']; cfg=config(pre/'resolved.yaml')
        for relative,expected in cp['small_sha256'].items():
            assert sha(Path(cp['path'])/'actor'/relative)==expected,'Frozen checkpoint changed'
        assert cfg['runner']['resume_dir']==cp['path']
        check=inspect_checkpoint(cp['path'],cfg,plan['repo'])
        assert check['step']==cp['step'] and check['contract_sha256']==cp['contract_sha256']
        assert dependency_snapshot(cfg)==row['dependencies'],'Initialization dependency changed'
        rt.mkdir(parents=True,mode=0o700)
        for name in ('resolved.yaml','environment.json'):
            dest=rt/name;assert not dest.exists();dest.write_bytes((pre/name).read_bytes());dest.chmod(0o600)
        argv=[plan['python'],'-u','-B',str(stage/'rlt_cycle.py'),'--cycle-dir',str(stage),'driver','--key',key]
        save(rt/'argv.json',argv);(rt/'command.txt').write_text(shlex.join(argv)+'\n')
        save(attempt,{'time':now(),'key':key,'namespace':row['namespace'],'release':release})
        environment=read(rt/'environment.json');assert not any(k in environment for k in MASKS)
        with (rt/'driver.log').open('x') as log:
            child=subprocess.Popen(argv,cwd=plan['repo'],env=environment,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=proc(child.pid);assert identity and identity['uid']==UID
        save(stage/(key+'-launched.json'),{'time':now(),'identity':identity,'namespace':row['namespace'],'run':row['new_run']})
        time.sleep(1)
        assert child.poll() is None,'Resume driver exited during launch: '+key
    update_watch(plan,stage)
    save(stage/'resumed-dispatched.json',{'time':now(),'cycle_id':stage.name,'release':release,
        'runs':{k:r['new_run'] for k,r in plan['runs'].items()},'status':'dispatched; real first-round validation pending'})
    return {'resumed_dispatched':True,'cycle_id':stage.name,'first_round_validation_pending':True}


def driver(stage,key):
    plan=load_plan(stage);row=plan['runs'][key];runtime=Path(row['new_run'])/'runtime'
    environment=read(runtime/'environment.json');assert not any(k in environment for k in MASKS)
    os.environ.update(environment)
    for path in reversed(environment.get('PYTHONPATH','').split(':')):
        if path:sys.path.insert(0,path)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(max(soft,min(4096,hard)),hard))
    from rlinf.scheduler import Cluster
    Cluster.NAMESPACE=row['namespace']
    save(runtime/'driver-identity.json',{**proc(os.getpid()),'namespace':row['namespace'],'time':now()})
    def terminate(signum,_frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    sys.argv=[str(Path(plan['repo'])/row['entry']),'--config-path',str(runtime),
        '--config-name','resolved','hydra.run.dir=.','hydra.output_subdir=null',
        'hydra.job.chdir=false','hydra/job_logging=stdout']
    exit_code=0;cleanup_error=None
    try:
        runpy.run_path(sys.argv[0],run_name='__main__')
    except SystemExit as exc:
        exit_code=exc.code if isinstance(exc.code,int) else 1
        raise
    except BaseException:
        exit_code=1
        raise
    finally:
        import ray
        signal.signal(signal.SIGUSR1,signal.SIG_IGN)
        try:
            if ray.is_initialized():
                job=ray.get_runtime_context().get_job_id();job=job.hex() if hasattr(job,'hex') else str(job)
                selected=active(actors(plan),row['namespace']);validate_actor_rows(selected,row['namespace'],{job})
                save(runtime/'cleanup-targets.json',{'time':now(),'job_id':job,'actors':selected})
                for actor in selected:
                    try:handle=ray.get_actor(actor['name'],namespace=row['namespace'])
                    except ValueError:continue
                    assert handle._actor_id.hex()==actor['actor_id']
                    ray.kill(handle,no_restart=True)
        except BaseException as exc:
            cleanup_error=type(exc).__name__+': '+str(exc)[:500]
            if exit_code==0:exit_code=1
        finally:
            ray.shutdown()
            save(runtime/'finished.json',{'time':now(),'exit_code':exit_code,'cleanup_error':cleanup_error})
            (runtime/'exit_code.txt').write_text(str(exit_code)+'\n')
        if cleanup_error and exit_code==1:raise RuntimeError(cleanup_error)


def status(stage):
    plan=load_plan(stage);result={'time':now(),'cycle_id':stage.name,'runs':{}}
    stopped=read(stage/'rlt-stopped.json') if (stage/'rlt-stopped.json').is_file() else None
    for key,row in plan['runs'].items():
        rt=Path(row['new_run'])/'runtime';identity=read(rt/'driver-identity.json') if (rt/'driver-identity.json').is_file() else None
        metrics=latest_metrics(row['new_run']);cp=stopped['runs'][key]['checkpoint'] if stopped else row['checkpoint']
        online=[v['value'] for k,v in metrics.items() if 'ready_for_online' in k]
        updates=[v['value'] for k,v in metrics.items() if 'update_step' in k]
        replay=[v['value'] for k,v in metrics.items() if 'global_min_replay_size' in k]
        critics=[v['value'] for k,v in metrics.items() if 'critic_updates_run' in k]
        rounds=[v['round'] for v in metrics.values()]
        log=''
        if (rt/'driver.log').is_file():
            with (rt/'driver.log').open('rb') as f:log=f.read(1048576).decode(errors='replace')
        healthy=bool(identity and same(identity) and online and online[-1]==1 and updates and updates[-1]>=cp['update_step']
            and replay and replay[-1]>=cp['replay_samples'] and rounds and max(rounds)>cp['step']
            and critics and critics[-1]>0
            and ('Resuming training from checkpoint directory '+cp['path']) in log)
        result['runs'][key]={'original_alive':same(row['original_identity']),'resume_identity':identity,
            'resume_alive':bool(identity and same(identity)),'checkpoint_step':cp['step'],'latest_metrics':metrics,
            'first_round_verified':healthy,'finished':read(rt/'finished.json') if (rt/'finished.json').is_file() else None}
    result['all_first_rounds_verified']=all(v['first_round_verified'] for v in result['runs'].values())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle-dir',required=True)
    parser.add_argument('action',choices=('prepare','stop','resume','status','driver'))
    parser.add_argument('--key',choices=('gpu4','gpu5','gpu6','gpu7'))
    parser.add_argument('--release-receipt')
    args=parser.parse_args();stage=checked_dir(args.cycle_dir)
    if args.action=='driver':
        assert args.key;driver(stage,args.key);return
    if args.action=='status':
        print(json.dumps(status(stage),ensure_ascii=False));return
    import fcntl
    stage.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (stage/'operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='prepare':result=prepare(stage)
        elif args.action=='stop':result=stop(stage)
        else:
            assert args.release_receipt;result=resume(stage,args.release_receipt)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
