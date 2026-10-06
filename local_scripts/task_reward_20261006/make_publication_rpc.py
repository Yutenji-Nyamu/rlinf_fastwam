"""Serialize the exact reviewed RM implementation and light evidence release.

Generate separate staging/push RPCs only; do not execute SSH or Git. Reuse the
isolated bell publication checkout. A stopped old WM owner is valid: verify its
frozen plan/source/evidence bytes, not its liveness. Also verify v1 and v2 RM
deployment bytes, including the already-published native launcher in each.
"""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parent
PREFIX = 'docs/world-model/task_reward_plan_20261006/'
CODE = 'local_scripts/task_reward_20261006/'
PRIOR = '462131e6b3182c86c4bb7ffafc7af5b4b62642fb'
RELEASE = 'task-reward-implementation-20261006-v1'
MANIFEST = PREFIX + 'implementation_manifest.json'
RUNTIME_FILES = (
    'rm_owner.py', 'rm_train.py', 'rm_inference.py', 'prepare_dataset.py',
    'test_task_reward.py', 'prepare_native_capture.py', 'validate_capture.py',
)
REUSED_LAUNCHER = 'local_scripts/rynn_binary_20261005/run_native_eval.py'
FILES = tuple(CODE + name for name in RUNTIME_FILES) + (
    CODE + 'rm_owner_v2.py', CODE + 'make_deploy_rpc.py', CODE + 'make_deploy_v2_rpc.py',
    CODE + 'make_publication_rpc.py', CODE + 'write_light_results.py',
    PREFIX + 'README.md', PREFIX + 'execution.md', PREFIX + 'light_results.json',
    PREFIX + 'published.json',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK_BINDINGS = r'''
def check_task_runtime():
    assert set(P['task_runtime_bindings'])=={'v1','v2'}
    for attempt,bindings in P['task_runtime_bindings'].items():
        target=S/('task-reward-'+attempt)
        assert target.is_dir() and not target.is_symlink() and target.stat().st_uid==20001
        owner_path=target/'run/owner-plan.json'
        owner=read(owner_path) if owner_path.exists() else None
        if owner is not None:
            assert owner['owner_dir']==str(target/'run') and owner['physical_gpus']==[4]
            expected_parent=O if attempt=='v1' else S/'task-reward-v1/run'
            assert owner['parent_owner']==str(expected_parent)
        for name,item in bindings.items():
            path=Path(item['remote_path'])
            assert path.parent==target/'code' and path.is_file() and not path.is_symlink()
            assert path.stat().st_uid==20001
            actual=digest(path.read_bytes())
            assert actual==item['sha256'],'RM published bytes differ from deployment: '+attempt+'/'+name
            if name in P['files']:
                assert actual==P['files'][name]['sha256']
            if owner is not None:
                assert owner['source_sha256'][str(path)]==actual,'RM owner did not freeze this source: '+attempt+'/'+name

'''


def main():
    initial = load_module('bell_initial', ROOT / 'local_scripts/wmrl_bell_20261005/make_publication_rpc.py')
    followup = load_module('bell_results', ROOT / 'local_scripts/wmrl_bell_20261005/make_followup_publication_rpc.py')
    prior = initial.prior_generator()
    published = json.loads((ROOT / (PREFIX + 'published.json')).read_text(encoding='utf-8-sig'))
    assert published['commit'] == PRIOR == published['verified_remote_commit']
    assert published['pushed'] is True and published['branch'] == initial.BRANCH
    runtime = json.loads((ROOT / 'docs/world-model/publication_bell_20261005/light_receipt.json').read_text(encoding='utf-8-sig'))
    prior.evidence_shape(runtime)
    assert runtime['ready_for_publication'] is True
    assert runtime['owner_dir'] == initial.S + '/runs/click-bell-v2'
    files = {}
    for name in FILES:
        prior.relative(name)
        path = ROOT / name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT), name
        raw = path.read_bytes()
        assert 0 < len(raw) <= 262144, name
        text = prior.scan(name, raw)
        if path.suffix == '.json':
            prior.evidence_shape(json.loads(text))
        if path.suffix == '.py':
            ast.parse(text, filename=name)
        files[name] = dict(bytes=len(raw), sha256=sha(raw), base64=base64.b64encode(raw).decode())
    task_bindings = {}
    for attempt in ('v1', 'v2'):
        attempt_files = {}
        for name in RUNTIME_FILES:
            published_name = CODE + ('rm_owner_v2.py' if attempt == 'v2' and name == 'rm_owner.py' else name)
            attempt_files[published_name] = dict(remote_path=initial.S + '/task-reward-' + attempt + '/code/' + name,
                sha256=files[published_name]['sha256'])
        task_bindings[attempt] = attempt_files
    launcher = ROOT / REUSED_LAUNCHER
    assert launcher.is_file() and not launcher.is_symlink()
    launcher_raw = launcher.read_bytes()
    prior.scan(REUSED_LAUNCHER, launcher_raw)
    ast.parse(launcher_raw, filename=REUSED_LAUNCHER)
    for attempt, attempt_files in task_bindings.items():
        attempt_files[REUSED_LAUNCHER] = dict(remote_path=initial.S + '/task-reward-' + attempt + '/code/run_native_eval.py',
            sha256=sha(launcher_raw))
    manifest = dict(schema_version=1, prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        parent=PRIOR, branch=initial.BRANCH, release_receipt_directory=initial.S + '/publication/' + RELEASE,
        files={name: {k: v for k, v in row.items() if k != 'base64'} for name, row in files.items()},
        task_runtime_bindings=task_bindings,
        scope='Task reward classifier implementation, exact GPU4 borrow/return owner, native capture and light execution evidence.',
        runtime_source_check='Pin earlier WM runtime/source/evidence without requiring its owner alive; verify RM deployment bytes and owner source bindings when present.',
        excluded=['weights/checkpoints', 'raw logs/media', 'full environments/plans', 'credentials',
                  'generated RPC payloads', 'copied reference sources', 'unrelated workspace changes'],
        committed=False, pushed=False,
        self_hash_rule='Manifest excludes itself; generated RPC pins its exact bytes.')
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    prior.scan(MANIFEST, raw)
    (ROOT / MANIFEST).write_bytes(raw)
    files[MANIFEST] = dict(bytes=len(raw), sha256=sha(raw), base64=base64.b64encode(raw).decode())
    payload = dict(prior_commit=PRIOR, branch=initial.BRANCH, files=files,
        secret_patterns=prior.SECRET_PATTERNS, manifest_path=MANIFEST, manifest_sha256=sha(raw),
        source_bindings=runtime['source_bindings'], evidence_pins=runtime['evidence_pins'],
        owner_plan_sha256=runtime['owner_plan_sha256'], runtime_repositories=runtime['runtime_repositories'],
        task_runtime_bindings=task_bindings)
    common = prior.REMOTE_COMMON
    for old, new in [
        ("BASE=S/'publication/opendw-v2'", "BASE=S/'publication/rynn-numeric-release-v1'"),
        ("W=S/'publication/rynn-success-release-v1'", "W=S/'publication/wmrl-bell-release-v1'"),
        ("E=S/'publication/rynn-success-release-20261005-v1'", "E=S/'publication/" + RELEASE + "'"),
        ("R=S/'rlinf-rynn-v1'", "R=S/'rlinf-opendw-bell-v1'"),
        ("O=S/'runs/rynn-success-v2'", "O=S/'runs/click-bell-v2'"),
        ("D=S/'rynn-control-v2'", "D=S/'click-bell-v2'")]:
        assert old in common
        common = common.replace(old, new)
    identities = initial.IDENTITIES.replace(initial.PRIOR, PRIOR)
    bindings = initial.BINDINGS.replace("P['files'][name]['sha256']", "item['sha256']")
    bindings = bindings.rstrip() + '\n    check_task_runtime()\n\n' + TASK_BINDINGS
    snapshot = initial.SNAPSHOT.replace("return {'repositories':snapshots,",
        "task_sources={attempt:{name:digest(Path(item['remote_path']).read_bytes()) for name,item in bindings.items()} for attempt,bindings in P['task_runtime_bindings'].items()}\n"
        "    task_owners={}\n"
        "    for attempt in P['task_runtime_bindings']:\n"
        "        owner_path=S/('task-reward-'+attempt)/'run/owner-plan.json'\n"
        "        task_owners[attempt]=digest(owner_path.read_bytes()) if owner_path.exists() else None\n"
        "    return {'task_runtime_source_sha256':task_sources,'task_owner_plan_sha256':task_owners,'repositories':snapshots,")
    assert snapshot != initial.SNAPSHOT
    common = initial.replace_function(common, 'identities', 'runtime_snapshot', identities)
    common = initial.replace_function(common, 'runtime_snapshot', 'remote_head', snapshot)
    common = initial.replace_function(common, 'check_source_bindings', 'staged_contract', bindings)
    stage = followup.STAGE.replace('publication/wmrl-bell-release-20261005-v1/published.json',
        'publication/wmrl-task-reward-plan-20261006-v1/published.json')
    stage = stage.replace('assert len(contents)==4', 'assert len(contents)==' + str(len(files)))
    push = prior.REMOTE_PUSH.replace('Add RynnValue reward adapters and semantic diagnostics',
        'Add task reward classifier training and native capture pipeline')
    old = "staged_contract(staged);assert not remote_head(),'Release branch appeared before this commit'"
    new = "staged_contract(staged);assert remote_head()==[P['prior_commit'],'refs/heads/'+P['branch']],'Published parent changed'"
    assert old in push
    push = push.replace(old, new)
    old = "if not remote:git(W,'push','personal','HEAD:refs/heads/'+P['branch'])"
    new = "if remote==[P['prior_commit'],'refs/heads/'+P['branch']]:git(W,'push','personal','HEAD:refs/heads/'+P['branch'])"
    assert old in push
    push = push.replace(old, new)
    header = '# Generated exact RM implementation publication; root reviews then executes.\nPAYLOAD = ' + repr(
        base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()) + '\n'
    outputs = {}
    for name, body in [('publication_stage_remote.py', stage), ('publication_push_remote.py', push)]:
        source = header + common + body
        ast.parse(source, filename=name)
        path = SCRIPT / name
        path.write_text(source, encoding='utf-8', newline='\n')
        outputs[name] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    print(json.dumps(dict(generator_only=True, files=len(files), exact_files=sorted(files),
        parent=PRIOR, branch=initial.BRANCH, scripts=outputs,
        receipt_directory=initial.S + '/publication/' + RELEASE,
        ssh=False, committed=False, pushed=False)))


if __name__ == '__main__':
    main()
