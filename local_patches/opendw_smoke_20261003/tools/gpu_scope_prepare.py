"""Prepare one GPU4 job's exact commname profile; never edits an existing profile."""
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
import xml.etree.ElementTree as ET

from gpu_scope_runtime import MANIFEST_ENV, digest, read_manifest


def inventory():
    xml = subprocess.check_output(["nvidia-smi", "-q", "-x"], timeout=30)
    root = ET.fromstring(xml)
    return root, [dict(index=i, uuid=g.findtext("uuid"), minor=int(g.findtext("minor_number")),
                       pci=g.findtext("./pci/pci_bus_id"),
                       free_mib=int(g.findtext("./fb_memory_usage/free").split()[0]),
                       contexts=[dict(pid=int(p.findtext("pid")), type=p.findtext("type"),
                                      memory=p.findtext("used_memory"))
                                 for p in g.findall("./processes/process_info")])
                  for i, g in enumerate(root.findall("gpu"))]


def exclusive(path, content, mode=0o600):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def profile_snapshot(driver_version):
    home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    roots = [home / ".nv/nvidia-application-profiles-rc",
             home / ".nv/nvidia-application-profiles-rc.d",
             Path("/etc/nvidia/nvidia-application-profiles-rc"),
             Path("/etc/nvidia/nvidia-application-profiles-rc.d"),
             Path(f"/usr/share/nvidia/nvidia-application-profiles-{driver_version}-rc")]
    found = {}
    for root in roots:
        if not root.exists():
            continue
        paths = sorted(root.iterdir()) if root.is_dir() else [root]
        for path in paths:
            if not path.is_file():
                continue
            if path.stat().st_size > 4 * 1024 * 1024:
                raise RuntimeError(f"Profile too large to audit: {path}")
            content = path.read_bytes()
            found[str(path)] = dict(sha256=digest(path), size=len(content))
            # Conservative precedence guard: do not silently compete with an
            # existing setting, even when it might be a harmless later rule.
            if b"EGLVisibleDGPUDevices" in content:
                raise RuntimeError(f"Existing EGL device rule needs a read-only precedence audit: {path}")
    return found, [str(root) for root in roots]


def prepare(args):
    if sys.platform != "linux" or os.getuid() == 0:
        raise RuntimeError("Prepare on the target Linux host as the ordinary training owner")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{7,63}", args.token):
        raise ValueError("Use a fresh 8-64 character ASCII job token")
    target = args.output.absolute()
    if target.exists():
        raise FileExistsError("Use a fresh scope directory; existing scopes are not replaced")
    if any(p.is_symlink() for p in [target, *target.parents]):
        raise RuntimeError("Scope directory must not traverse symlinks")
    root, cards = inventory()
    gpu = cards[4]
    if not gpu["uuid"].startswith("GPU-") or not 0 <= gpu["minor"] < 32:
        raise RuntimeError("Cannot identify GPU4 UUID/minor")
    previous, search_paths = profile_snapshot(root.findtext("driver_version"))
    account = pwd.getpwuid(os.getuid())
    home = Path(account.pw_dir)
    profile_dir = home / ".nv/nvidia-application-profiles-rc.d"
    for p in [home / ".nv", profile_dir]:
        if p.is_symlink():
            raise RuntimeError(f"Profile directory must not be a symlink: {p}")
        p.mkdir(mode=0o700, exist_ok=True)
        info = p.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise RuntimeError(f"Profile directory is not privately owned: {p}")
    target.mkdir(parents=True, mode=0o700)
    bootstrap = target / "bootstrap"
    bootstrap.mkdir(mode=0o700)
    receipts = target / "receipts"
    receipts.mkdir(mode=0o700)
    basename = f"libopendw_scope_g4_{args.token}.so"
    marker = target / basename
    source = target / "marker.c"
    exclusive(source, b"int opendw_gpu_scope_marker(void) { return 1; }\n")
    subprocess.run([args.cc, "-shared", "-fPIC", "-nostdlib", f"-Wl,-soname,{basename}",
                    "-o", str(marker), str(source)], check=True, timeout=60)
    marker.chmod(0o500)
    tools = Path(__file__).resolve().parent
    boot = bootstrap / "sitecustomize.py"
    runtime = bootstrap / "gpu_scope_runtime.py"
    exclusive(boot, (tools / "gpu_scope_bootstrap/sitecustomize.py").read_bytes())
    exclusive(runtime, (tools / "gpu_scope_runtime.py").read_bytes())
    profile = profile_dir / f"00-opendw-g4-{args.token}.json"
    commname = "odw4-" + hashlib.sha256(args.token.encode()).hexdigest()[:10]
    rule = {"rules": [{"pattern": {"feature": "commname", "matches": commname},
                       "profile": ["EGLVisibleDGPUDevices", 1 << gpu["minor"]]}]}
    exclusive(profile, (json.dumps(rule, indent=2) + "\n").encode())
    for name, old in previous.items():
        if digest(name) != old["sha256"]:
            raise RuntimeError(f"Existing profile changed during preparation: {name}")
    manifest = dict(schema=3, token=args.token, commname=commname,
                    home=account.pw_dir, user=account.pw_name, title_api="ray._raylet.setproctitle",
                    profile_feature="commname", uid=os.getuid(), hostname=socket.gethostname(),
                    physical_gpu=4, gpu_uuid=gpu["uuid"], pci=gpu["pci"], minor=gpu["minor"],
                    mask=1 << gpu["minor"], driver_version=root.findtext("driver_version"),
                    receipts_dir=str(receipts), existing_profiles=previous, profile_search_paths=search_paths)
    for key, path in dict(marker=marker, profile=profile, bootstrap=boot, runtime=runtime).items():
        manifest[key + "_path"] = str(path)
        manifest[key + "_sha256"] = digest(path)
    path = target / "scope.json"
    exclusive(path, (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps(dict(manifest=str(path), gpu_uuid=gpu["uuid"], profile=str(profile), commname=commname,
                          prepared=True, gpu_probe_completed=False)))


def environment_fragment(manifest_path, inherited=None, *, probe=False):
    m = read_manifest(manifest_path)
    account = pwd.getpwuid(os.getuid())
    if m.get("home") != account.pw_dir or m.get("user") != account.pw_name:
        raise RuntimeError("Scope must record the actual owner's HOME and USER")
    env = os.environ if inherited is None else inherited
    marker = m["marker_path"]
    preload = env.get("LD_PRELOAD", "")
    paths = [p for p in re.split(r"[:\s]+", preload) if p and p != marker]
    if any(Path(p).name.startswith("libopendw_scope_") for p in paths):
        raise RuntimeError("Do not combine two job scope markers in one process")
    bootstrap = str(Path(m["bootstrap_path"]).parent)
    pythonpath = [p for p in env.get("PYTHONPATH", "").split(":") if p and p != bootstrap]
    fragment = {MANIFEST_ENV: str(Path(manifest_path).resolve(strict=True)),
                "LD_PRELOAD": ":".join([marker, *paths]),
                "PYTHONPATH": ":".join([bootstrap, *pythonpath]),
                "__GL_APPLICATION_PROFILE": "1", "__GL_APPLICATION_PROFILE_LOG": "1",
                "HOME": account.pw_dir, "USER": account.pw_name, "LOGNAME": account.pw_name}
    # Driver and CPU management actors retain the physical hardware inventory;
    # RLinf supplies GPU4 placement to actual actor/env/rollout workers.
    if probe:
        fragment["CUDA_VISIBLE_DEVICES"] = "4"
    return fragment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--token", required=True)
    create.add_argument("--cc", default="cc")
    env = sub.add_parser("env")
    env.add_argument("--manifest", type=Path, required=True)
    env.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        print(json.dumps(environment_fragment(args.manifest, probe=args.probe), indent=2))


if __name__ == "__main__":
    main()
