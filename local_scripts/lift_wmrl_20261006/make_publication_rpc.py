"""Generate reviewed lift-pot WMRL publication RPCs; never execute SSH or Git.

Reuse the bell branch's isolated checkout and the established separate stage /
push transaction. Publish an explicit source/evidence allowlist only. Loss PNG
is the sole permitted binary; model weights and raw example media stay remote.
"""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREFIX = "docs/world-model/task_reward_plan_20261006/"
CODE = "local_scripts/lift_wmrl_20261006/"
PRIOR = "c6b73f0792083fda17c5b7610b7629a7b25ae53c"
RELEASE = "lift-wmrl-integration-20261006-v1"
MANIFEST = PREFIX + "integration_manifest.json"
LIGHT = PREFIX + "integration_light_receipt.json"
PNG = PREFIX + "lift_pot_rm_loss.png"
RUNTIME_FILES = (
    "rm_adapter.py", "patch_lift.py", "test_rm_adapter.py", "reset_prepare.py",
    "native_seed_prepare.py", "recipe_prepare.py", "lifecycle_prepare.py",
    "assemble_owner.py", "native_prechecks.py",
)
FILES = tuple(CODE + name for name in RUNTIME_FILES) + (
    CODE + "make_prepare_rpc.py", CODE + "make_curve.py", CODE + "error_frames_remote.py",
    CODE + "make_publication_rpc.py", CODE + "make_assemble_rpc.py",
    CODE + "launch_remote.py", CODE + "status_remote.py", CODE + "initial_scores_remote.py",
    "local_scripts/task_reward_20261006/rm_inference.py",
    PREFIX + "integration_20261006.md", PREFIX + "lift_pot_rm_history.csv", PNG,
    PREFIX + "error_review_metadata.json", PREFIX + "initial_reward_scores.json",
    PREFIX + "README.md", PREFIX + "results_published.json", LIGHT,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IDENTITIES = r'''def identities():
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
    assert P['prior_commit']=='c6b73f0792083fda17c5b7610b7629a7b25ae53c'
    assert P['branch']=='codex/wmrl-bell-reward-20261005'
    for path in (S,BASE,R,D,S/'rlinf-rynn-binary-v1'):
        assert path.is_dir() and path.stat().st_uid==20001 and path.resolve().is_relative_to(S.resolve())
    assert W.resolve().is_relative_to((S/'publication').resolve())
    assert E.resolve().is_relative_to((S/'publication').resolve())
    assert git(BASE,'remote','get-url','personal').decode().strip()=='git@github.com:Yutenji-Nyamu/rlinf_fastwam.git'
'''

SNAPSHOT = r'''def runtime_snapshot():
    snapshots={}
    for name,path in [('wm',R),('native',S/'rlinf-rynn-binary-v1')]:
        head=git(path,'rev-parse','HEAD').decode().strip()
        assert head==EXPECTED_RUNTIME_HEAD
        snapshots[name]={'path':str(path),'head':head,
            'status_sha256':digest(git(path,'status','--porcelain','--untracked-files=all')),
            'diff_sha256':digest(git(path,'diff','HEAD','--'))}
    plan_path=D/'prepared/owner-plan.json'
    plan=read(plan_path)
    assert plan['owner_dir']==str(O) and plan['physical_gpus']==[4,5,6,7]
    assert plan['repo']==str(R) and plan['repo_head']==EXPECTED_RUNTIME_HEAD
    frozen={}
    for name,expected in plan['source_sha256'].items():
        path=Path(name)
        if path.suffix not in ('.py','.yaml'):
            continue
        assert path.resolve().is_relative_to(S.resolve()) and path.stat().st_uid==20001
        actual=digest(path.read_bytes());assert actual==expected,'Frozen source changed: '+name
        frozen[name]=actual
    deployed={name:digest(Path(item['remote_path']).read_bytes()) for name,item in P['source_bindings'].items()}
    return {'repositories':snapshots,'prepared_owner_plan_sha256':digest(plan_path.read_bytes()),
        'frozen_source_sha256':frozen,'deployed_source_sha256':deployed}
'''

BINDINGS = r'''def check_source_bindings():
    for name,item in P['source_bindings'].items():
        path=Path(item['remote_path'])
        assert path.is_absolute() and path.parent==D/'code' and path.is_file() and not path.is_symlink()
        assert path.stat().st_uid==20001
        actual=digest(path.read_bytes())
        assert actual==item['sha256']==P['files'][name]['sha256'],'Published source differs from deployment: '+name
    plan=read(D/'prepared/owner-plan.json')
    assert plan['provenance']['policy_starts_from_original_sft'] is True
    assert plan['budget']['task']=='lift_pot'
    assert plan['budget']['reward']['source']=='single_task_resnet_classifier'
'''

DECODED = r'''def decoded_files():
    files={}
    for name,row in P['files'].items():
        relative(name);data=base64.b64decode(row['base64'],validate=True)
        assert len(data)==row['bytes'] and digest(data)==row['sha256'] and 0<len(data)<=262144
        if name==P['allowed_png']:
            assert data.startswith(b'\x89PNG\r\n\x1a\n')
        else:
            assert Path(name).suffix in ('.py','.md','.json','.csv')
            text=data.decode('utf-8-sig')
            for expression in P['secret_patterns']:assert not re.search(expression,text),'Secret-like content: '+name
        files[name]=data
    assert P['manifest_path'] in files and digest(files[P['manifest_path']])==P['manifest_sha256']
    return files
'''


def main():
    initial = load("lift_initial_publication", ROOT / "local_scripts/wmrl_bell_20261005/make_publication_rpc.py")
    followup = load("lift_followup_publication", ROOT / "local_scripts/wmrl_bell_20261005/make_followup_publication_rpc.py")
    prior = initial.prior_generator()
    receipt = json.loads((ROOT / (PREFIX + "results_published.json")).read_text(encoding="utf-8-sig"))
    assert receipt["commit"] == receipt["verified_remote_commit"] == PRIOR and receipt["pushed"] is True
    assert receipt["branch"] == initial.BRANCH
    light = json.loads((ROOT / LIGHT).read_text(encoding="utf-8-sig"))
    prior.evidence_shape(light)
    assert light["ready_for_publication"] is True
    files = {}
    for name in FILES:
        prior.relative(name)
        path = ROOT / name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT), name
        raw = path.read_bytes()
        assert 0 < len(raw) <= 262144, name
        if name == PNG:
            assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        else:
            text = prior.scan(name, raw)
            if path.suffix == ".py":
                ast.parse(text, filename=name)
            if path.suffix == ".json":
                prior.evidence_shape(json.loads(text))
        files[name] = dict(bytes=len(raw), sha256=sha(raw), base64=base64.b64encode(raw).decode())
    bindings = {CODE + name: dict(remote_path=initial.S + "/lift-pot-v1/code/" + name,
        sha256=files[CODE + name]["sha256"]) for name in RUNTIME_FILES}
    inference = "local_scripts/task_reward_20261006/rm_inference.py"
    bindings[inference] = dict(remote_path=initial.S + "/lift-pot-v1/code/rm_inference.py", sha256=files[inference]["sha256"])
    manifest = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
        parent=PRIOR, branch=initial.BRANCH, release_receipt_directory=initial.S + "/publication/" + RELEASE,
        files={name: {key: value for key, value in row.items() if key != "base64"} for name, row in files.items()},
        source_bindings=bindings, execution_evidence=LIGHT,
        scope="Lift-pot RM convergence, error review metadata, reset/native seeds, B16 integration and lifecycle preparation.",
        execution_claim="Dated integration document and light receipt state execution progress; source publication alone does not establish smoke/formal success.",
        excluded=["weights/checkpoints", "reset arrays", "raw logs/videos/error images", "full environments and owner plans",
                  "credentials", "generated RPC payloads", "copied references", "unrelated shared HANDOFF changes"],
        runtime_source_check="Exact uploaded source bindings; prepared owner-frozen Python/YAML; two runtime repository snapshots before/after publication.",
        committed=False, pushed=False, self_hash_rule="Manifest excludes itself; generated payload pins its exact bytes.")
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    prior.scan(MANIFEST, raw)
    (ROOT / MANIFEST).write_bytes(raw)
    files[MANIFEST] = dict(bytes=len(raw), sha256=sha(raw), base64=base64.b64encode(raw).decode())
    payload = dict(prior_commit=PRIOR, branch=initial.BRANCH, files=files, source_bindings=bindings,
        secret_patterns=prior.SECRET_PATTERNS, manifest_path=MANIFEST, manifest_sha256=sha(raw), allowed_png=PNG)
    common = prior.REMOTE_COMMON
    for old, new in [
        ("BASE=S/'publication/opendw-v2'", "BASE=S/'publication/rynn-numeric-release-v1'"),
        ("W=S/'publication/rynn-success-release-v1'", "W=S/'publication/wmrl-bell-release-v1'"),
        ("E=S/'publication/rynn-success-release-20261005-v1'", "E=S/'publication/" + RELEASE + "'"),
        ("R=S/'rlinf-rynn-v1'", "R=S/'rlinf-opendw-bell-v1'"),
        ("O=S/'runs/rynn-success-v2'", "O=S/'runs/lift-pot-v1'"),
        ("D=S/'rynn-control-v2'", "D=S/'lift-pot-v1'")]:
        assert old in common
        common = common.replace(old, new)
    common = initial.replace_function(common, "identities", "runtime_snapshot", IDENTITIES)
    common = initial.replace_function(common, "runtime_snapshot", "remote_head", SNAPSHOT)
    common = initial.replace_function(common, "decoded_files", "check_source_bindings", DECODED)
    common = initial.replace_function(common, "check_source_bindings", "staged_contract", BINDINGS)
    stage = followup.STAGE.replace("publication/wmrl-bell-release-20261005-v1/published.json", "publication/task-reward-results-20261006-v1/published.json")
    stage = stage.replace("assert len(contents)==4", "assert len(contents)==" + str(len(files)))
    push = prior.REMOTE_PUSH.replace("Add RynnValue reward adapters and semantic diagnostics", "Integrate converged lift-pot reward model with existing WMRL pipeline")
    old = "staged_contract(staged);assert not remote_head(),'Release branch appeared before this commit'"
    new = "staged_contract(staged);assert remote_head()==[P['prior_commit'],'refs/heads/'+P['branch']],'Published parent changed'"
    assert old in push
    push = push.replace(old, new)
    old = "if not remote:git(W,'push','personal','HEAD:refs/heads/'+P['branch'])"
    new = "if remote==[P['prior_commit'],'refs/heads/'+P['branch']]:git(W,'push','personal','HEAD:refs/heads/'+P['branch'])"
    assert old in push
    push = push.replace(old, new)
    header = "# Generated exact lift-pot integration publication; root reviews then executes.\nPAYLOAD = " + repr(base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()) + "\n"
    outputs = {}
    for name, body in (("publication_stage_remote.py", stage), ("publication_push_remote.py", push)):
        source = header + common + body
        ast.parse(source, filename=name)
        path = HERE / name
        path.write_text(source, encoding="utf-8", newline="\n")
        outputs[name] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    print(json.dumps(dict(generator_only=True, files=len(files), exact_files=sorted(files), parent=PRIOR,
        branch=initial.BRANCH, receipt_directory=initial.S + "/publication/" + RELEASE,
        scripts=outputs, ssh=False, committed=False, pushed=False)))


if __name__ == "__main__":
    main()
