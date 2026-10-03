"""Read-only scope evidence. No CUDA/EGL/Vulkan/SAPIEN import or device creation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from gpu_scope_prepare import environment_fragment, inventory
from gpu_scope_runtime import digest, read_manifest, verify_ray_title_scope

SAFE_ENV = ("HOME", "USER", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "PYTHONPATH",
            "RLINF_OPENDW_GPU_SCOPE_MANIFEST", "__GL_APPLICATION_PROFILE",
            "__GL_APPLICATION_PROFILE_LOG", "__EGL_VENDOR_LIBRARY_FILENAMES",
            "__EGL_VENDOR_LIBRARY_DIRS", "VK_ICD_FILENAMES", "VK_DRIVER_FILES",
            "DISPLAY", "XDG_CONFIG_HOME", "EGL_DEVICE_ID", "MUJOCO_EGL_DEVICE_ID")


def process_evidence():
    maps = Path("/proc/self/maps").read_text().splitlines()
    matching = [line for line in maps if any(text in line for text in (
        "libopendw_scope_", "libGL", "libEGL", "libnvidia", "libcuda", "vulkan", "sapien"))]
    return dict(pid=os.getpid(), uid=os.getuid(), comm=Path("/proc/self/comm").read_text().strip(),
                exe=os.readlink("/proc/self/exe"), environment={k: os.environ.get(k) for k in SAFE_ENV},
                selected_maps=matching,
                gpu_modules_imported=[k for k in sys.modules if k.startswith(("torch", "sapien", "cupy"))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    m = read_manifest(args.manifest)
    if args.child:
        title_proof = verify_ray_title_scope(m)
        evidence = process_evidence()
        if evidence['gpu_modules_imported'] or os.environ.get('CUDA_VISIBLE_DEVICES') != '4':
            raise RuntimeError('CPU probe imported GPU modules or changed numeric CUDA visibility')
        print(json.dumps(dict(**evidence, ray_title_proof=title_proof)))
        return
    profiles = []
    for name in m["profile_search_paths"]:
        root = Path(name)
        if not root.exists():
            profiles.append(dict(path=str(root), exists=False))
            continue
        for path in sorted(root.iterdir()) if root.is_dir() else [root]:
            if not path.is_file():
                continue
            raw = path.read_text(errors="replace")
            info = path.stat()
            profiles.append(dict(path=str(path), realpath=str(path.resolve()), uid=info.st_uid,
                                 mode=oct(info.st_mode & 0o777), sha256=digest(path),
                                 content=raw if "EGLVisibleDGPUDevices" in raw or str(path) == m["profile_path"] else None))
    globals_path = Path.home() / ".nv/nvidia-application-profile-globals-rc"
    root, cards = inventory()
    result = dict(manifest_path=str(args.manifest), manifest=m, parent=process_evidence(),
                  profile_search_order=profiles,
                  global_profile=globals_path.read_text() if globals_path.is_file() else None,
                  driver_version=root.findtext("driver_version"),
                  kernel_driver=Path("/proc/driver/nvidia/version").read_text(),
                  gpu_identity=[{k: v for k, v in row.items() if k != "contexts"} for row in cards],
                  receipts=[])
    for path in sorted(Path(m["receipts_dir"]).glob("*.json"), key=lambda p: p.stat().st_mtime)[-40:]:
        result["receipts"].append(dict(path=str(path), data=json.loads(path.read_text())))
    if args.probe_output:
        result["probe_files"] = {}
        for name in ("report.json", "camera.json", "child.json", "probe.log"):
            path = args.probe_output / name
            result["probe_files"][name] = path.read_text(errors="replace")[-16000:] if path.is_file() else None
    if shutil.which("readelf"):
        result["marker_elf"] = subprocess.run(["readelf", "-h", "-d", "-l", m["marker_path"]],
                                             text=True, capture_output=True, timeout=15).stdout
    env = dict(os.environ)
    env.update(environment_fragment(args.manifest, env, probe=True))
    child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                            "--manifest", str(args.manifest.resolve()), "--child"],
                           env=env, text=True, capture_output=True, timeout=30)
    result["fresh_cpu_child"] = dict(exit_code=child.returncode, stdout=child.stdout, stderr=child.stderr)
    result["scope"] = "Read-only process/profile/ELF/NVML audit; no GPU API or renderer loaded"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
