"""Build exact Rynn publication scripts; no local Git, SSH, or runtime mutation.

Run after the execution document and optional sanitized receipts are ready.
The generated stage script creates a separate publication worktree; the push
script publishes only that reviewed staged diff, never the training checkout.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[2]
PRIOR = '07faf0636163b8ca6a604f94eb8cea74705a125a'
SOURCE_BRANCH = 'codex/robotwin-opendw-wmrl-20261003'
BRANCH = 'codex/rynnvalue-wmrl-release-20261005'
PATCH = 'local_patches/rynn_wmrl_20261005/'
OPS = 'local_scripts/rynn_wmrl_20261005/'
DOC = 'docs/world-model/robotwin_pipeline_20261003/'
EVIDENCE = 'docs/world-model/publication_rynn_20261005/'
MANIFEST = EVIDENCE + 'source_manifest.json'
PATCH_FILES = [PATCH + name for name in (
    'rlinf/envs/world_model/opendw_robotwin_env.py',
    'rlinf/envs/world_model/rynn_success.py',
    'rlinf/envs/world_model/rynn_actor_mask.py',
    'tools/rynn_success_service.py', 'tools/opendw_service_batched.py',
    'tools/patch_actor_rynn_mask.py', 'tools/test_rynn_success_service.py',
    'tests/test_env_and_mask.py')]
OP_FILES = [OPS + name for name in (
    'rynn_formal_owner.py', 'rynn_handoff.py', 'rynn_gate_client.py', 'test_rynn_owner.py',
    'rynn_formal_owner_v2.py', 'prepare_v2_remote.py', 'launch_v2_remote.py', 'test_rynn_owner_v2.py',
    'rynn_diagnostic.py', 'rynn_diagnostic_owner.py', 'test_rynn_diagnostic_owner.py',
    'preflight_rynn_diagnostic.py', 'build_diag_stage_rpc.py', 'prepare_diagnostic_samples_remote.py',
    'prepare_diagnostic_owner_remote.py', 'launch_diagnostic_remote.py', 'diagnostic_status_remote.py',
    'collect_evidence_remote.py', 'collect_diagnostic_evidence_remote.py',
    'native_status_remote.py', 'prepare_native_frames_remote.py', 'verify_official_git_remote.py',
    'rynn_native_diagnostic.py', 'rynn_diagnostic_owner_v2.py', 'test_rynn_diagnostic_owner_v2.py',
    'build_native_stage_rpc.py', 'prepare_native_owner_remote.py', 'launch_native_remote.py',
    'prepare_remote.py', 'prepare_samples_remote.py', 'prepare_plan_remote.py',
    'validate_assets_remote.py', 'launch_handoff_remote.py', 'inventory_remote.py',
    'status_remote.py', 'build_stage_rpc.py', 'build_v2_stage_rpc.py', 'make_publication_rpc.py')]
DOC_FILES = [DOC + name for name in (
    'rynnvalue_integration_plan_20261005.md', 'rynnvalue_batch_audit_20261005.md',
    'rynnvalue_execution_20261005.md', 'reward_saturation_discussion_20261005.md',
    'candidate_comparison_20261005.md', 'general_reward_integration_20261003.md',
    'rynnvalue_semantic_diagnosis_20261005.md',
    'rynnvalue_source_receipt_20261005.json')]
FORBIDDEN_KEYS = {'password', 'passwd', 'ssh_password', 'api_key', 'access_token', 'refresh_token',
    'secret_key', 'authorization', 'token', 'environment', 'environment_json', 'env_vars'}
SECRET_PATTERNS = [
    r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    r'\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}\b', r'\bAKIA[0-9A-Z]{16}\b',
    r'''(?i)authorization\s*[:=]\s*['"]?Bearer\s+[A-Za-z0-9._~-]{16,}''',
    r'''(?i)https?://[^\s/@:'"]+:[^\s/@'"]+@''',
    r'(?i)[?&](?:X-Amz-Signature|X-Goog-Signature|Signature)=[A-Za-z0-9%+/=]{24,}',
    r'''(?i)\b(?:password|passwd|ssh_password|api_key|access_token|secret_key)\b\s*[:=]\s*['"][^'"\r\n]{4,}['"]''',
    r'(?i)\bzxcv\d{4}(?:zxcv\d{4})?\b',
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and path.parts
    assert not any(part in ('..', '.', '.git', '__pycache__') for part in path.parts)
    assert not any(char in name for char in ('\\', '\0', ':'))
    return path


def scan(name, data):
    text = data.decode('utf-8-sig')
    for pattern in SECRET_PATTERNS:
        assert not re.search(pattern, text), 'Secret-like content: ' + name
    return text


def evidence_shape(value):
    if isinstance(value, dict):
        for key, child in value.items():
            assert str(key).lower() not in FORBIDDEN_KEYS, 'Do not publish full environment/credential fields: ' + str(key)
            evidence_shape(child)
    elif isinstance(value, list):
        for child in value:
            evidence_shape(child)


REMOTE_COMMON = r'''
import base64
from datetime import datetime,timezone
import hashlib,json,os,re,socket,subprocess
from pathlib import Path,PurePosixPath

P=json.loads(base64.b64decode(PAYLOAD,validate=True))
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
BASE=S/'publication/opendw-v2'
W=S/'publication/rynn-success-release-v1'
E=S/'publication/rynn-success-release-20261005-v1'
R=S/'rlinf-rynn-v1'
O=S/'runs/rynn-success-v2'
D=S/'rynn-control-v2'
DD=S/'rynn-diagnosis-v1'
ND=S/'rynn-diagnosis-v2'
EXPECTED_RUNTIME_HEAD='2151a08ee1bd75df1bef0d8190e594bd5c7f7977'

def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
def record(path,value):
    with Path(path).open('xb') as f:f.write(encoded(value));f.flush();os.fsync(f.fileno())
def git(repo,*args):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0',GIT_LFS_SKIP_SMUDGE='1')
    return subprocess.run(['git','-c','gc.auto=0','-c','core.autocrlf=false','-c','core.safecrlf=false',
        '-C',str(repo),*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=120).stdout
def relative(name):
    p=PurePosixPath(name)
    assert not p.is_absolute() and p.parts and not any(x in ('.','..','.git','__pycache__') for x in p.parts)
    assert not any(x in name for x in ('\\','\0',':'))
    return p
def identities():
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
    assert P['prior_commit']=='07faf0636163b8ca6a604f94eb8cea74705a125a'
    assert P['branch']=='codex/rynnvalue-wmrl-release-20261005'
    for path in (S,BASE,R,O,D,DD,ND):
        assert path.is_dir() and path.stat().st_uid==20001 and path.resolve().is_relative_to(S.resolve())
    assert W.resolve().is_relative_to((S/'publication').resolve())
    assert E.resolve().is_relative_to((S/'publication').resolve())
    assert git(BASE,'remote','get-url','personal').decode().strip()=='git@github.com:Yutenji-Nyamu/rlinf_fastwam.git'
def runtime_snapshot():
    plan=read(O/'owner-plan.json')
    selected={p:sha for p,sha in plan['source_sha256'].items()
              if Path(p).is_relative_to(D) or Path(p).is_relative_to(R)}
    for path,sha in selected.items():assert digest(Path(path).read_bytes())==sha,'Frozen running source changed: '+path
    head=git(R,'rev-parse','HEAD').decode().strip()
    assert head==EXPECTED_RUNTIME_HEAD,'Running checkout HEAD changed'
    diag=read(DD/'run/owner-plan.json')
    diag_sources={p:sha for p,sha in diag['source_sha256'].items() if Path(p).is_relative_to(DD/'code')}
    for path,sha in diag_sources.items():assert digest(Path(path).read_bytes())==sha,'Frozen diagnostic source changed: '+path
    native=read(ND/'run/owner-plan.json')
    native_sources={p:sha for p,sha in native['source_sha256'].items() if Path(p).is_relative_to(ND/'code')}
    for path,sha in native_sources.items():assert digest(Path(path).read_bytes())==sha,'Frozen native diagnostic source changed: '+path
    return {'head':head,'status_sha256':digest(git(R,'status','--porcelain','--untracked-files=all')),
        'diff_sha256':digest(git(R,'diff','HEAD','--')),
        'owner_plan_sha256':digest((O/'owner-plan.json').read_bytes()),'frozen_sources':selected,
        'diagnostic_owner_plan_sha256':digest((DD/'run/owner-plan.json').read_bytes()),
        'diagnostic_sources':diag_sources,
        'native_owner_plan_sha256':digest((ND/'run/owner-plan.json').read_bytes()),
        'native_sources':native_sources}
def remote_head():
    return git(BASE,'ls-remote','--heads','personal','refs/heads/'+P['branch']).decode().split()
def decoded_files():
    files={}
    for name,row in P['files'].items():
        relative(name);data=base64.b64decode(row['base64'],validate=True)
        assert len(data)==row['bytes'] and digest(data)==row['sha256'] and len(data)<=262144
        for expression in P['secret_patterns']:assert not re.search(expression,data.decode('utf-8-sig')),'Secret-like content: '+name
        files[name]=data
    assert P['manifest_path'] in files and digest(files[P['manifest_path']])==P['manifest_sha256']
    return files
def check_source_bindings():
    plan=read(O/'owner-plan.json');frozen=plan['source_sha256']
    entry=P['runtime_entrypoint']
    assert entry==str(D/'code/rynn_formal_owner_v2.py')
    assert plan['owner_script_sha256']==digest(Path(entry).read_bytes()),'Wrong live owner entrypoint'
    for name,path in P['runtime_bindings'].items():
        assert frozen.get(path)==P['files'][name]['sha256'],'Publication differs from owner-frozen code: '+name
        assert digest(Path(path).read_bytes())==P['files'][name]['sha256']
    v2final=read(O/'final.json')
    assert v2final['terminal_status']=='failed' and v2final['recovery_error'] is None
    assert v2final['rlt_return_dispatched'] is True,'V2 must remain documented as failed and recovered'
    # The single-card diagnosis is a separate transaction, not a formal restart.
    diag=read(DD/'run/owner-plan.json');dfrozen=dict(diag['source_sha256'])
    assert diag['mode']=='rynn-single-gpu-diagnostic' and diag['physical_gpus']==[4]
    tests_path=DD/'prepared/cpu-tests.json'
    assert dfrozen[str(tests_path)]==digest(tests_path.read_bytes())
    tests=read(tests_path);assert tests['all_cpu_tests_passed'] is True
    dfrozen.update(tests['source_sha256'])
    assert P['diagnostic_entrypoint']==str(DD/'code/rynn_diagnostic_owner.py')
    for name,path in P['diagnostic_bindings'].items():
        assert Path(path).is_relative_to(DD/'code')
        assert dfrozen.get(path)==P['files'][name]['sha256'],'Diagnostic differs from tested/frozen code: '+name
        assert digest(Path(path).read_bytes())==P['files'][name]['sha256']
    native=read(ND/'run/owner-plan.json');nfrozen=dict(native['source_sha256'])
    assert native['mode']=='rynn-single-gpu-diagnostic' and native['physical_gpus']==[4]
    assert native['parent_owner']==str(DD/'run')
    tests_path=ND/'prepared/cpu-tests.json'
    assert nfrozen[str(tests_path)]==digest(tests_path.read_bytes())
    tests=read(tests_path);assert tests['all_cpu_tests_passed'] is True
    nfrozen.update(tests['source_sha256'])
    assert P['native_entrypoint']==str(ND/'code/rynn_diagnostic_owner_v2.py')
    for name,path in P['native_bindings'].items():
        assert Path(path).is_relative_to(ND/'code')
        assert nfrozen.get(path)==P['files'][name]['sha256'],'Native diagnostic differs from tested/frozen code: '+name
        assert digest(Path(path).read_bytes())==P['files'][name]['sha256']
    # Historical v1 owner/handoff are retained as provenance, not live entrypoints.
    v1=read(S/'runs/rynn-success-v1/owner-plan.json')['source_sha256']
    for name,path in P['archive_bindings'].items():
        assert v1.get(path)==P['files'][name]['sha256'],'V1 archive differs from frozen failure source: '+name
        assert digest(Path(path).read_bytes())==P['files'][name]['sha256']
def staged_contract(staged):
    assert git(W,'branch','--show-current').decode().strip()==P['branch']
    names={x.decode() for x in git(W,'diff','--cached','--name-only','-z').split(b'\0') if x}
    assert names==set(staged['staged_names']) and names.issubset(P['files']) and names
    assert digest(git(W,'diff','--cached','--binary'))==staged['staged_diff_sha256']
    assert not git(W,'diff','--name-only').strip() and not git(W,'ls-files','--others','--exclude-standard').strip()
    for name,row in P['files'].items():
        assert digest((W/name).read_bytes())==row['sha256']
        assert digest(git(W,'show',':'+name))==row['sha256']
'''

REMOTE_STAGE = r'''
def main():
    identities()
    assert not W.exists() and not W.is_symlink() and not E.exists(),'Do not replay a publication stage'
    assert not remote_head(),'New release branch already exists remotely'
    old_remote=git(BASE,'ls-remote','--heads','personal','refs/heads/'+P['source_branch']).decode().split()
    assert old_remote==[P['prior_commit'],'refs/heads/'+P['source_branch']],'Published base changed; review source commit'
    git(BASE,'cat-file','-e',P['prior_commit']+'^{commit}')
    before=runtime_snapshot();check_source_bindings();contents=decoded_files()
    E.mkdir(mode=0o700)
    try:
        git(BASE,'worktree','add','-b',P['branch'],str(W),P['prior_commit'])
        assert git(W,'rev-parse','HEAD').decode().strip()==P['prior_commit']
        assert not git(W,'status','--porcelain','--untracked-files=all').strip()
        for name,data in contents.items():
            path=W.joinpath(*relative(name).parts)
            assert path.resolve().is_relative_to(W.resolve()) and not path.is_symlink()
            assert not path.exists() or path.is_file()
            for parent in path.parents:
                if parent==W:break
                assert not parent.is_symlink()
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            assert path.read_bytes()==data
        git(W,'add','-f','--',*sorted(contents))
        staged_names=[x.decode() for x in git(W,'diff','--cached','--name-only','-z').split(b'\0') if x]
        assert staged_names and set(staged_names).issubset(contents)
        for name,data in contents.items():assert git(W,'show',':'+name)==data,'Git filters changed '+name
        assert not git(W,'diff','--name-only').strip() and not git(W,'ls-files','--others','--exclude-standard').strip()
        git(W,'-c','core.whitespace=cr-at-eol','diff','--cached','--check')
        diff=git(W,'diff','--cached','--binary');stat=git(W,'diff','--cached','--stat')
        after=runtime_snapshot();assert after==before,'Running sources changed during publication staging'
        receipt={'time':datetime.now(timezone.utc).isoformat(),'status':'staged_for_review',
            'prior_commit':P['prior_commit'],'branch':P['branch'],'manifest_sha256':P['manifest_sha256'],
            'files':{name:{'bytes':len(raw),'sha256':digest(raw)} for name,raw in contents.items()},
            'staged_names':staged_names,'staged_diff_sha256':digest(diff),'runtime_before':before,'runtime_after':after,
            'committed':False,'pushed':False,'runtime_unchanged':True}
        record(E/'staged.json',receipt);(E/'staged.diff').write_bytes(diff);(E/'staged.stat.txt').write_bytes(stat)
        print(json.dumps({'status':'staged_for_review','worktree':str(W),'receipt':str(E/'staged.json'),
            'files':len(contents),'changed_files':len(staged_names),'diff_sha256':digest(diff),'runtime_unchanged':True}))
        print(stat.decode())
    except BaseException as exc:
        record(E/'failed.json',{'error_type':type(exc).__name__,'error':str(exc),'publication_worktree_retained':str(W),
            'committed':False,'pushed':False})
        raise
if __name__=='__main__':main()
'''

REMOTE_PUSH = r'''
def main():
    identities()
    assert not (E/'published.json').exists(),'Publication is already recorded'
    staged=read(E/'staged.json')
    assert staged['prior_commit']==P['prior_commit'] and staged['manifest_sha256']==P['manifest_sha256']
    assert staged['status']=='staged_for_review' and staged['committed'] is False and staged['pushed'] is False
    assert runtime_snapshot()==staged['runtime_after'],'Running source snapshot changed; re-review publication'
    check_source_bindings()
    if (E/'committed.json').exists():
        created=read(E/'committed.json');commit=created['commit']
        assert created['branch']==P['branch'] and created['manifest_sha256']==P['manifest_sha256']
        assert git(W,'rev-parse','HEAD').decode().strip()==commit
        assert git(W,'rev-parse','HEAD^').decode().strip()==P['prior_commit']
        assert not git(W,'status','--porcelain','--untracked-files=all').strip()
        for name,row in P['files'].items():assert digest(git(W,'show','HEAD:'+name))==row['sha256']
    else:
        assert git(W,'rev-parse','HEAD').decode().strip()==P['prior_commit']
        staged_contract(staged);assert not remote_head(),'Release branch appeared before this commit'
        git(W,'commit','-m','Add RynnValue reward adapters and semantic diagnostics')
        commit=git(W,'rev-parse','HEAD').decode().strip()
        assert git(W,'rev-parse','HEAD^').decode().strip()==P['prior_commit']
        assert not git(W,'status','--porcelain','--untracked-files=all').strip()
        created={'time':datetime.now(timezone.utc).isoformat(),'prior_commit':P['prior_commit'],
            'commit':commit,'branch':P['branch'],'manifest_sha256':P['manifest_sha256'],'pushed':False}
        record(E/'committed.json',created)
    remote=remote_head()
    if not remote:git(W,'push','personal','HEAD:refs/heads/'+P['branch'])
    else:assert remote==[commit,'refs/heads/'+P['branch']],'Remote branch differs; never force-push'
    assert remote_head()==[commit,'refs/heads/'+P['branch']]
    assert runtime_snapshot()==staged['runtime_after'],'Running sources changed during publication'
    receipt=dict(created,pushed=True,verified_remote_commit=commit,files=len(P['files']),runtime_unchanged=True)
    record(E/'published.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', action='append', default=[])
    args = parser.parse_args()
    assert len(args.evidence) == len(set(args.evidence)) <= 4
    for name in args.evidence:
        path = relative(name)
        assert path.parent == PurePosixPath(EVIDENCE) and path.suffix == '.json' and name != MANIFEST
    files, bindings, archives, diagnostics, natives = {}, {}, {}, {}, {}
    server = '/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/'
    for name in PATCH_FILES + OP_FILES + DOC_FILES + args.evidence:
        relative(name)
        path = ROOT / name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        data = path.read_bytes()
        assert len(data) <= 262144
        text = scan(name, data)
        if name.endswith('.py'):
            ast.parse(text, filename=name)
        if name in args.evidence or name.endswith('source_receipt_20261005.json'):
            value = json.loads(text)
            assert isinstance(value, dict)
            evidence_shape(value)
        files[name] = dict(source_local=name, bytes=len(data), sha256=digest(data), base64=base64.b64encode(data).decode())
        if name.startswith(PATCH + 'rlinf/'):
            bindings[name] = server + 'rlinf-rynn-v1/' + name[len(PATCH):]
        elif name in PATCH_FILES:
            bindings[name] = server + 'rynn-control-v2/code/' + Path(name).name
        elif Path(name).name in ('rynn_formal_owner_v2.py', 'prepare_v2_remote.py', 'launch_v2_remote.py',
                                 'test_rynn_owner_v2.py', 'rynn_gate_client.py', 'test_rynn_owner.py'):
            bindings[name] = server + 'rynn-control-v2/code/' + Path(name).name
        elif Path(name).name in ('rynn_formal_owner.py', 'rynn_handoff.py'):
            archives[name] = server + 'rynn-control-v1/code/' + Path(name).name
        elif Path(name).name in ('rynn_diagnostic.py', 'rynn_diagnostic_owner.py', 'test_rynn_diagnostic_owner.py', 'preflight_rynn_diagnostic.py'):
            diagnostics[name] = server + 'rynn-diagnosis-v1/code/' + Path(name).name
        elif Path(name).name in ('rynn_native_diagnostic.py', 'rynn_diagnostic_owner_v2.py', 'test_rynn_diagnostic_owner_v2.py'):
            natives[name] = server + 'rynn-diagnosis-v2/code/' + Path(name).name
    entrypoint = server + 'rynn-control-v2/code/rynn_formal_owner_v2.py'
    diagnostic_entrypoint = server + 'rynn-diagnosis-v1/code/rynn_diagnostic_owner.py'
    native_entrypoint = server + 'rynn-diagnosis-v2/code/rynn_diagnostic_owner_v2.py'
    manifest = dict(schema_version=1, created_time_utc=datetime.now(timezone.utc).isoformat(),
        prior_remote_commit=PRIOR, source_branch=SOURCE_BRANCH, branch=BRANCH,
        purpose='RynnValue BF16 repair, sparse reward/unknown masks, failed V2 semantic gate, and separate GPU4 input diagnosis',
        files={name: {key: value for key, value in row.items() if key != 'base64'} for name, row in files.items()},
        runtime_bindings=bindings, runtime_entrypoint=entrypoint, archive_bindings=archives,
        diagnostic_bindings=diagnostics, diagnostic_entrypoint=diagnostic_entrypoint,
        native_bindings=natives, native_entrypoint=native_entrypoint,
        archived_v1_notice='V1 failed before training because its floating-point value head was not converted to BF16; v1 owner/handoff files are retained as history, not active launch instructions.',
        v2_execution_notice='V2 passed dtype/API execution but rejected all 16 expert samples as No at its semantic gate. It failed before training and returned RLT. The GPU4 diagnostic is an independent run, not resumed formal WMRL.',
        native_execution_notice='The native-input comparison is a separate GPU4-only diagnostic. Its engineering and semantic results are defined by the final listed receipt; source inclusion makes no success claim.',
        execution_evidence=args.evidence,
        execution_claim='Only the explicitly listed receipts establish progress; source publication does not establish smoke or formal success.',
        excluded=['running checkout Git/index mutation', 'full owner plans/environments/tokens', 'raw logs/TensorBoard/media',
                  'weights/checkpoints/reset datasets', 'generated upload RPC/base64 duplication', 'shared HANDOFF and unrelated dirty'],
        self_hash_rule='Manifest excludes itself; generated RPC payload pins its exact bytes.')
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    scan(MANIFEST, raw)
    (ROOT / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / MANIFEST).write_bytes(raw)
    files[MANIFEST] = dict(source_local=MANIFEST, bytes=len(raw), sha256=digest(raw), base64=base64.b64encode(raw).decode())
    payload = dict(prior_commit=PRIOR, source_branch=SOURCE_BRANCH, branch=BRANCH, files=files,
        secret_patterns=SECRET_PATTERNS, manifest_path=MANIFEST, manifest_sha256=digest(raw), runtime_bindings=bindings,
        runtime_entrypoint=entrypoint, archive_bindings=archives,
        diagnostic_bindings=diagnostics, diagnostic_entrypoint=diagnostic_entrypoint,
        native_bindings=natives, native_entrypoint=native_entrypoint)
    header = '# Generated isolated Rynn publication transaction.\nPAYLOAD = ' + repr(
        base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()) + '\n'
    results = {}
    for name, body in (('publication_stage_remote.py', REMOTE_STAGE), ('publication_push_remote.py', REMOTE_PUSH)):
        source = header + REMOTE_COMMON + body
        ast.parse(source, filename=name)
        path = Path(__file__).with_name(name)
        path.write_text(source, encoding='utf-8', newline='\n')
        results[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
    print(json.dumps(dict(generator_only=True, files=len(files), evidence=len(args.evidence), runtime_bindings=len(bindings),
        diagnostic_bindings=len(diagnostics), native_bindings=len(natives),
        scripts=results, manifest=str(ROOT / MANIFEST), ssh=False, committed=False, pushed=False)))


if __name__ == '__main__':
    main()
