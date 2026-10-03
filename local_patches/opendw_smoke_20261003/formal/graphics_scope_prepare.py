"""Stage privately, then activate GPU4-7 profiles only after an exact handoff.

No profile is installed by prepare_stage. activate records intent and active
state before the caller's native probe. Source/runtime/manifest stay immutable;
separate receipts describe activation, rollback, or a native-probe failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import socket
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from graphics_scope_runtime import (
    LEGACY_MANIFEST_ENV, MANIFEST_ENV, PHYSICAL_GPUS, digest, owned_file, read_manifest,
)


def read(path):
    return json.loads(Path(path).read_text())


def exclusive(path, content, mode=0o600):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def record(path, data):
    exclusive(path, (json.dumps(data, indent=2) + "\n").encode())


def current_boot():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def require_linux_owner():
    if sys.platform != "linux" or os.getuid() == 0:
        raise RuntimeError("Use the ordinary training owner on the target Linux host")


def inventory():
    root = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], timeout=30))
    cards = []
    for index, row in enumerate(root.findall("gpu")):
        cards.append(dict(physical_gpu=index, gpu_uuid=row.findtext("uuid"),
                          minor=int(row.findtext("minor_number")), pci=row.findtext("./pci/pci_bus_id")))
    if len(cards) != 8:
        raise RuntimeError("This deployment expects the explicitly audited eight-GPU host")
    return root.findtext("driver_version"), cards


def validate_existing_device_profile(content):
    """Only disjoint private comm rules; never import an account-wide fallback."""
    data = json.loads(content)
    if set(data) != {"rules"} or not isinstance(data["rules"], list) or not data["rules"]:
        raise RuntimeError("Existing device profile needs a separate precedence review")
    names = []
    for rule in data["rules"]:
        if set(rule) != {"pattern", "profile"}:
            raise RuntimeError("Unexpected fields in existing device rule")
        pattern = rule["pattern"]
        if not isinstance(pattern, dict) or set(pattern) != {"feature", "matches"}:
            raise RuntimeError("Unconditional/composite device rules require a separate review")
        name = pattern["matches"]
        if pattern["feature"] != "commname" or not isinstance(name, str):
            raise RuntimeError("Only explicit private commname device rules can coexist")
        if not re.fullmatch(r"(?:odw4-[0-9a-f]{10}|odwf[4567]-[0-9a-f]{8})", name):
            raise RuntimeError("Unrecognized private scope commname")
        setting = rule["profile"]
        if (not isinstance(setting, list) or len(setting) != 2 or setting[0] != "EGLVisibleDGPUDevices"
                or type(setting[1]) is not int or setting[1] <= 0 or setting[1] & (setting[1] - 1)):
            raise RuntimeError("Existing private scope must have exactly one device bit")
        names.append(name)
    return names


def profile_snapshot(driver_version, audited_device_profiles):
    account = pwd.getpwuid(os.getuid())
    home = Path(account.pw_dir)
    roots = [home / ".nv/nvidia-application-profiles-rc", home / ".nv/nvidia-application-profiles-rc.d",
             Path("/etc/nvidia/nvidia-application-profiles-rc"), Path("/etc/nvidia/nvidia-application-profiles-rc.d"),
             Path(f"/usr/share/nvidia/nvidia-application-profiles-{driver_version}-rc")]
    audited = {str(Path(path).resolve(strict=True)) for path in audited_device_profiles}
    found, device_paths, names = {}, set(), set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.iterdir()) if root.is_dir() else [root]:
            if not path.is_file():
                continue
            if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
                raise RuntimeError(f"Profile needs a separate path/size review: {path}")
            content = path.read_bytes()
            found[str(path)] = dict(sha256=digest(path), size=len(content))
            if b"EGLVisibleDGPUDevices" not in content:
                continue
            owned_file(path)
            if str(path.resolve()) not in audited:
                raise RuntimeError(f"Device profile was not explicitly audited: {path}")
            device_paths.add(str(path.resolve()))
            for name in validate_existing_device_profile(content):
                if name in names:
                    raise RuntimeError("Overlapping old private scope names")
                names.add(name)
    if device_paths != audited:
        raise RuntimeError("Audited device profile list differs from the active search paths")
    return found, [str(path) for path in roots], sorted(names)


def prepare_stage(output, token, audited_device_profiles, cc="cc", inherited=None):
    """CPU preparation only: the draft profile stays inside output."""
    require_linux_owner()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{7,63}", token):
        raise ValueError("Use a fresh 8-64 character token")
    output = Path(output).absolute()
    if output.exists() or any(path.is_symlink() for path in [output, *output.parents]):
        raise RuntimeError("Use a fresh canonical scope directory without symlink ancestors")
    version, cards = inventory()
    previous, search_paths, names = profile_snapshot(version, audited_device_profiles)
    account = pwd.getpwuid(os.getuid())
    profile_dir = Path(account.pw_dir) / ".nv/nvidia-application-profiles-rc.d"
    # Do not create or change any account graphics path in this staging step.
    if not profile_dir.is_dir() or profile_dir.is_symlink():
        raise RuntimeError("Expected the existing private graphics profile directory")
    for directory in (profile_dir.parent, profile_dir):
        info = directory.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise RuntimeError("Graphics profile directory is not privately owned")
    output.mkdir(parents=True, mode=0o700)
    bootstrap = output / "bootstrap"
    bootstrap.mkdir(mode=0o700)
    (output / "receipts").mkdir(mode=0o700)
    source = output / "marker.c"
    marker = output / f"libopendw_formal_scope_{token}.so"
    exclusive(source, b"int opendw_formal_scope_marker(void) { return 1; }\n")
    subprocess.run([cc, "-shared", "-fPIC", "-nostdlib", f"-Wl,-soname,{marker.name}",
                    "-o", str(marker), str(source)], check=True, timeout=60)
    marker.chmod(0o500)
    tools = Path(__file__).resolve().parent
    runtime = bootstrap / "graphics_scope_runtime.py"
    boot = bootstrap / "sitecustomize.py"
    exclusive(runtime, (tools / "graphics_scope_runtime.py").read_bytes())
    exclusive(boot, (tools / "graphics_scope_bootstrap.py").read_bytes())
    selected, rules = {}, []
    suffix = hashlib.sha256(token.encode()).hexdigest()[:8]
    for gpu in PHYSICAL_GPUS:
        card = dict(cards[gpu])
        if not card["gpu_uuid"].startswith("GPU-") or not 0 <= card["minor"] < 32:
            raise RuntimeError("Invalid GPU UUID/minor inventory")
        card.update(mask=1 << card["minor"], commname=f"odwf{gpu}-{suffix}")
        if card["commname"] in names:
            raise RuntimeError("Fresh token collides with an existing private profile")
        selected[str(gpu)] = card
        rules.append({"pattern": {"feature": "commname", "matches": card["commname"]},
                      "profile": ["EGLVisibleDGPUDevices", card["mask"]]})
    draft = output / "graphics-profile.staged.json"
    exclusive(draft, (json.dumps({"rules": rules}, indent=2) + "\n").encode())
    profile = profile_dir / f"00-opendw-formal-{token}.json"
    if profile.exists() or profile.is_symlink():
        raise RuntimeError("Fresh profile destination already exists")
    manifest = dict(schema=1, token=token, uid=os.getuid(), user=account.pw_name, home=account.pw_dir,
                    hostname=socket.gethostname(), boot_id=current_boot(), driver_version=version,
                    physical_gpus=list(PHYSICAL_GPUS), cpu_full_mask_target=4, cards=selected,
                    profile_path=str(profile), profile_sha256=digest(draft), staged_profile_path=str(draft),
                    receipts_dir=str(output / "receipts"), existing_profiles=previous,
                    profile_search_paths=search_paths, audited_device_profiles=sorted(str(Path(p).resolve()) for p in audited_device_profiles))
    for key, path in dict(marker=marker, runtime=runtime, bootstrap=boot).items():
        manifest[key + "_path"] = str(path)
        manifest[key + "_sha256"] = digest(path)
    record(output / "scope.json", manifest)
    fragment = _environment_fragment(manifest, output / "scope.json", inherited or {})
    record(output / "environment-fragment.json", fragment)
    result = dict(time=time.time(), status="staged_only", global_profile_installed=False,
                  manifest=str(output / "scope.json"), manifest_sha256=digest(output / "scope.json"),
                  profile=str(profile), profile_sha256=manifest["profile_sha256"],
                  environment_fragment_file=str(output / "environment-fragment.json"),
                  environment_fragment_sha256=digest(output / "environment-fragment.json"))
    record(output / "staged.json", result)
    return result


def staged_manifest(scope_dir):
    require_linux_owner()
    root = Path(scope_dir).resolve(strict=True)
    path = owned_file(root / "scope.json")
    manifest = read(path)
    if (manifest["uid"] != os.getuid() or manifest["hostname"] != socket.gethostname()
            or manifest["boot_id"] != current_boot() or manifest["physical_gpus"] != list(PHYSICAL_GPUS)):
        raise RuntimeError("Staged scope host/identity changed")
    staged = read(owned_file(root / "staged.json"))
    if staged["manifest_sha256"] != digest(path):
        raise RuntimeError("Staged manifest changed")
    owned_file(root / "environment-fragment.json", staged["environment_fragment_sha256"])
    for key in ("marker", "runtime", "bootstrap"):
        owned_file(manifest[key + "_path"], manifest[key + "_sha256"])
    owned_file(manifest["staged_profile_path"], manifest["profile_sha256"])
    expected = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".nv/nvidia-application-profiles-rc.d" / f"00-opendw-formal-{manifest['token']}.json"
    if Path(manifest["profile_path"]) != expected or expected.parent.is_symlink():
        raise RuntimeError("Staged profile destination changed")
    return root, manifest


def identity_alive(identity):
    try:
        path = Path("/proc") / str(int(identity["pid"]))
        tail = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return (path.stat().st_uid == int(identity["uid"]) and int(tail[19]) == int(identity["start"])
                and tail[0] not in ("Z", "X"))
    except FileNotFoundError:
        return False


def scoped_processes(manifest_path=None):
    """Read only our own process env; races exit cleanly, unreadable live PIDs fail."""
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            tail = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if tail[0] in ("Z", "X"):
                continue
            entries = (path / "environ").read_bytes().split(b"\0")
            env = dict(item.split(b"=", 1) for item in entries if b"=" in item)
            scopes = {key: os.fsdecode(env[key.encode()]) for key in (LEGACY_MANIFEST_ENV, MANIFEST_ENV)
                      if env.get(key.encode())}
            if scopes and (manifest_path is None or str(manifest_path) in scopes.values()):
                found.append(dict(pid=int(path.name), uid=os.getuid(), start=int(tail[19]), scopes=scopes))
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise RuntimeError(f"Cannot establish scope retirement for live owned PID {path.name}") from exc
    return found


def verify_retirement(proof_path):
    proof = read(owned_file(proof_path))
    if (proof.get("schema") != 1 or proof.get("uid") != os.getuid()
            or proof.get("hostname") != socket.gethostname() or proof.get("boot_id") != current_boot()):
        raise RuntimeError("Retirement proof host/identity differs")
    owner = proof["previous_owner"]
    if owner.get("uid") != os.getuid() or identity_alive(owner):
        raise RuntimeError("Previous WM owner has not reached its exact terminal handoff")
    final_record = proof["previous_owner_final"]
    final = read(owned_file(final_record["path"], final_record["sha256"]))
    if final.get("terminal_status") not in ("completed", "failed"):
        raise RuntimeError("Previous owner has no conclusive terminal record")
    release_record = proof["rlt_stop_receipt"]
    release = read(owned_file(release_record["path"], release_record["sha256"]))
    if (release.get("gpus_released") != list(PHYSICAL_GPUS)
            or release.get("all_original_drivers_stopped") is not True
            or release.get("all_original_namespaces_empty") is not True):
        raise RuntimeError("Four-card RLT stop receipt is incomplete")
    remaining = scoped_processes()
    if remaining:
        raise RuntimeError("Old scoped processes remain; do not install new profiles: " + json.dumps(remaining))
    return dict(proof_path=str(Path(proof_path).resolve()), proof_sha256=digest(proof_path),
                previous_owner=owner, previous_owner_final=final_record, rlt_stop_receipt=release_record,
                no_scoped_processes=True, checked_at=time.time())


def activation_payload(root, manifest, retirement, receipt_path):
    fragment_path = root / "environment-fragment.json"
    return dict(time=time.time(), status="active", scope_id=manifest["token"], uid=os.getuid(),
                hostname=socket.gethostname(), boot_id=current_boot(), manifest=str(root / "scope.json"),
                manifest_sha256=digest(root / "scope.json"), profile=manifest["profile_path"],
                profile_sha256=manifest["profile_sha256"], native_probe_verified=False,
                runtime_path=manifest["runtime_path"], runtime_sha256=manifest["runtime_sha256"],
                environment_fragment=read(fragment_path), environment_fragment_file=str(fragment_path),
                environment_fragment_sha256=digest(fragment_path),
                activation_receipt=str(receipt_path), retirement=retirement)


def activate(scope_dir, retirement_proof, activation_receipt):
    """Install exactly this new profile and record activation before native probe.

    A same-intent retry can recover a crash between profile creation and receipt;
    it never overwrites a profile or an existing receipt.
    """
    root, manifest = staged_manifest(scope_dir)
    receipt = Path(activation_receipt).absolute()
    if receipt.exists():
        active = read(owned_file(receipt))
        if active.get("status") != "active" or active.get("manifest_sha256") != digest(root / "scope.json"):
            raise RuntimeError("Conflicting activation receipt")
        read_manifest(root / "scope.json")
        return active
    retirement = verify_retirement(retirement_proof)
    version, _ = inventory()
    if version != manifest["driver_version"]:
        raise RuntimeError("Driver version changed since staging")
    destination = Path(manifest["profile_path"])
    intent_path = root / "activation-intent.json"
    expected_intent = dict(manifest_sha256=digest(root / "scope.json"),
                           profile=str(destination), profile_sha256=manifest["profile_sha256"],
                           activation_receipt=str(receipt), retirement_proof_sha256=digest(retirement_proof))
    if intent_path.exists():
        if read(owned_file(intent_path)) != expected_intent:
            raise RuntimeError("Inspect the prior activation intent; this is a different handoff")
    else:
        if destination.exists() or destination.is_symlink():
            raise RuntimeError("Unowned pre-existing profile; do not adopt it")
        record(intent_path, expected_intent)
    # Audit before creation. Recovery after a completed exact write accepts only
    # that profile; all previously frozen device profiles must remain unchanged.
    for path, details in manifest["existing_profiles"].items():
        if digest(path) != details["sha256"]:
            raise RuntimeError("Existing graphics rules changed after staging: " + path)
    if not destination.exists():
        current, _, _ = profile_snapshot(version, manifest["audited_device_profiles"])
        if current != manifest["existing_profiles"]:
            raise RuntimeError("Graphics search paths changed after staging")
        exclusive(destination, Path(manifest["staged_profile_path"]).read_bytes())
    owned_file(destination, manifest["profile_sha256"])
    # Mark active immediately. Even if later runtime/probe verification fails,
    # callers must return RLT using this scope or perform a proven exact rollback.
    active = activation_payload(root, manifest, retirement, receipt)
    record(receipt, active)
    read_manifest(root / "scope.json")
    return active


def rollback_exact(scope_dir, activation_receipt, rollback_receipt):
    """Optional explicit rollback after all users of the new scope have exited."""
    root, manifest = staged_manifest(scope_dir)
    active = read(owned_file(activation_receipt))
    if active.get("status") != "active" or active.get("manifest_sha256") != digest(root / "scope.json"):
        raise RuntimeError("Rollback requires this exact active scope receipt")
    if scoped_processes(str(root / "scope.json")):
        raise RuntimeError("New scope is still in use; do not remove its profile")
    receipt = Path(rollback_receipt)
    if receipt.exists():
        raise RuntimeError("Do not replay rollback; inspect its existing receipt")
    destination = owned_file(manifest["profile_path"], manifest["profile_sha256"])
    record(root / "rollback-intent.json", dict(time=time.time(), profile=str(destination),
            profile_sha256=manifest["profile_sha256"], activation_receipt_sha256=digest(activation_receipt)))
    destination.unlink()
    result = dict(time=time.time(), status="rolled_back", profile_removed=True,
                  profile=str(destination), manifest_sha256=digest(root / "scope.json"),
                  activation_receipt_sha256=digest(activation_receipt))
    record(receipt, result)
    return result


def _environment_fragment(manifest, manifest_path, env, *, probe=False, physical_gpu=6):
    account = pwd.getpwuid(os.getuid())
    if manifest["home"] != account.pw_dir or manifest["user"] != account.pw_name:
        raise RuntimeError("Scope HOME/USER mismatch")
    if env.get(LEGACY_MANIFEST_ENV):
        raise RuntimeError("Caller must explicitly retire the old GPU4 fragment before merging a new one")
    marker = manifest["marker_path"]
    preload = [part for part in re.split(r"[:\s]+", env.get("LD_PRELOAD", "")) if part and part != marker]
    if any(Path(part).name.startswith(("libopendw_scope_", "libopendw_formal_scope_")) for part in preload):
        raise RuntimeError("Do not combine two private scope markers")
    bootstrap = str(Path(manifest["bootstrap_path"]).parent)
    pythonpath = [part for part in env.get("PYTHONPATH", "").split(":") if part and part != bootstrap]
    fragment = {MANIFEST_ENV: str(Path(manifest_path).resolve(strict=True)),
                "LD_PRELOAD": ":".join([marker, *preload]), "PYTHONPATH": ":".join([bootstrap, *pythonpath]),
                "__GL_APPLICATION_PROFILE": "1", "__GL_APPLICATION_PROFILE_LOG": "1",
                "HOME": account.pw_dir, "USER": account.pw_name, "LOGNAME": account.pw_name}
    if probe:
        if physical_gpu not in PHYSICAL_GPUS:
            raise RuntimeError("Native probe must use one explicit authorized physical card")
        fragment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    return fragment


def staged_environment_fragment(scope_dir, inherited=None, *, probe=False, physical_gpu=6):
    """Only builds values for a frozen future launch; does not enable this scope."""
    root, manifest = staged_manifest(scope_dir)
    return _environment_fragment(manifest, root / "scope.json", inherited or {},
                                 probe=probe, physical_gpu=physical_gpu)


def environment_fragment(manifest_path, inherited=None, *, probe=False, physical_gpu=6):
    """Validate the active profile; no CVD override for RLinf discovery."""
    manifest = read_manifest(manifest_path)
    return _environment_fragment(manifest, manifest_path, os.environ if inherited is None else inherited,
                                 probe=probe, physical_gpu=physical_gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--token", required=True)
    stage.add_argument("--audited-device-profile", action="append", type=Path, default=[])
    stage.add_argument("--cc", default="cc")
    use = sub.add_parser("activate")
    use.add_argument("--scope-dir", type=Path, required=True)
    use.add_argument("--retirement-proof", type=Path, required=True)
    use.add_argument("--activation-receipt", type=Path, required=True)
    env = sub.add_parser("env")
    env.add_argument("--manifest", type=Path, required=True)
    env.add_argument("--probe", action="store_true")
    env.add_argument("--physical-gpu", type=int, default=6)
    args = parser.parse_args()
    if args.command == "stage":
        result = prepare_stage(args.output, args.token, args.audited_device_profile, args.cc)
    elif args.command == "activate":
        result = activate(args.scope_dir, args.retirement_proof, args.activation_receipt)
    else:
        result = environment_fragment(args.manifest, probe=args.probe, physical_gpu=args.physical_gpu)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
