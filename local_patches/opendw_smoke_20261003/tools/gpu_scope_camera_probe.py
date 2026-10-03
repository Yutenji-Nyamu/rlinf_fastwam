"""GPU4-only three-frame SAPIEN probe; no policy, task actions, Ray mutation, or training."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpu_scope_prepare import environment_fragment, exclusive, inventory
from gpu_scope_runtime import bind_renderer, check_marker, owned_file, read_manifest, verify_cuda_mapping, verify_ray_title_scope


def contexts(pid):
    return [dict(gpu=g["index"], uuid=g["uuid"], **p)
            for g in inventory()[1] for p in g["contexts"] if p["pid"] == pid]


def save(path, data):
    exclusive(path, (json.dumps(data, indent=2) + "\n").encode())


def identity(pid):
    path = Path(f"/proc/{pid}")
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, uid=path.stat().st_uid, start=int(fields[19]))


def baseline_for_probe(args, cards):
    current = {row["pid"] for row in cards[4]["contexts"]}
    if args.allow_existing_owned_pids is None:
        if current:
            raise RuntimeError("GPU4 must be empty unless an explicit owned identity list is provided")
        return []
    if os.getuid() != 20001:
        raise RuntimeError("Shared GPU4 probe exception is restricted to UID20001")
    supplied = json.loads(owned_file(args.allow_existing_owned_pids).read_text())
    if not isinstance(supplied, list):
        raise ValueError("Owned PID file must be an array of {pid, uid, start} identities")
    allowed = {}
    for row in supplied:
        item = {key: int(row[key]) for key in ("pid", "uid", "start")}
        if item["uid"] != 20001 or item["pid"] in allowed:
            raise ValueError("Baseline UID must be 20001 and PID identities must be unique")
        if identity(item["pid"]) != item:
            raise RuntimeError(f"Baseline identity changed: {item['pid']}")
        allowed[item["pid"]] = item
    if current != set(allowed):
        raise RuntimeError(f"GPU4 PID set differs from explicit baseline: observed={sorted(current)}, allowed={sorted(allowed)}")
    if cards[4]["free_mib"] < 16 * 1024:
        raise RuntimeError("Shared GPU4 probe requires at least 16 GiB free VRAM")
    return list(allowed.values())


def child(args):
    m = read_manifest(args.manifest)
    check_marker(m)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4" or os.environ.get("DISPLAY"):
        raise RuntimeError("Probe requires numeric GPU4 visibility and headless execution")
    # Stress both direct comm changes and Ray's public process-title setter.
    libc = C.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [C.c_int, C.c_void_p, C.c_ulong, C.c_ulong, C.c_ulong]
    name = C.create_string_buffer(b"ray::EnvWorker")
    if libc.prctl(15, C.cast(name, C.c_void_p), 0, 0, 0) != 0:
        raise OSError(C.get_errno(), "Cannot emulate Ray comm rename")
    if Path("/proc/self/comm").read_text().strip() != "ray::EnvWorker":
        raise RuntimeError("Ray-like comm rename failed")
    title_proof = verify_ray_title_scope(m)
    check_marker(m)
    # Verify logical CUDA0 maps to the planned physical PCI bus without creating
    # a policy/tensor allocation or probing a different device.
    mapping = verify_cuda_mapping(m)
    import numpy as np
    import sapien
    from PIL import Image
    binding = bind_renderer(m)
    observed = contexts(os.getpid())
    if any(row["gpu"] != 4 for row in observed):
        raise RuntimeError(f"Out-of-scope context after import: {observed}")
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")
    scene = sapien.Scene()
    scene.set_timestep(1 / 250)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [1, 1, 1])
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[0.2, 0.2, 0.2], material=[0.9, 0.1, 0.1])
    actor = builder.build_static(name="opendw_scope_probe_box")
    camera = scene.add_camera("scope_probe", 64, 64, 0.8, 0.1, 10)
    camera.entity.set_pose(sapien.Pose([-1, 0, 0]))
    frames = []
    try:
        for frame in range(3):
            scene.step()
            scene.update_render()
            camera.take_picture()
            rgba = camera.get_picture("Color")
            rgb = rgba[..., :3]
            if (rgba.shape != (64, 64, 4) or not np.isfinite(rgba).all()
                    or float(rgb.max()) <= 0.02 or float(rgb.std()) <= 0.01):
                raise RuntimeError("Invalid or blank real-camera frame")
            frames.append(dict(index=frame, shape=list(rgba.shape),
                               minimum=float(rgb.min()), maximum=float(rgb.max()), std=float(rgb.std())))
        Image.fromarray((rgb.clip(0, 1) * 255).astype(np.uint8)).save(args.output / "rgb.png")
        observed = contexts(os.getpid())
        if not observed or any(row["gpu"] != 4 for row in observed):
            raise RuntimeError(f"Probe contexts exceed GPU4 or are missing: {observed}")
        save(args.output / "camera.json", dict(passed=True, pid=os.getpid(), frames=frames,
             contexts=observed, binding=binding, cuda_mapping=mapping, ray_title_proof=title_proof,
             comm=Path("/proc/self/comm").read_text().strip(), real_ray_worker=False))
    finally:
        scene.clear()
        del camera, actor, scene


def parent(args):
    if sys.platform != "linux":
        raise RuntimeError("Run the GPU probe on the target Linux server")
    m = read_manifest(args.manifest)
    _, cards = inventory()
    if cards[4]["uuid"] != m["gpu_uuid"] or cards[4]["minor"] != m["minor"]:
        raise RuntimeError("Current GPU4 UUID/minor changed")
    baseline = baseline_for_probe(args, cards)
    baseline_pids = {row["pid"] for row in baseline}
    baseline_contexts = [dict(gpu=g["index"], uuid=g["uuid"], **p)
                         for g in cards for p in g["contexts"] if p["pid"] in baseline_pids]
    if args.output.exists():
        raise FileExistsError("Use a fresh probe output directory")
    args.output.mkdir(parents=True, mode=0o700)
    env = dict(os.environ)
    env.update(environment_fragment(args.manifest, env, probe=True))
    env.pop("DISPLAY", None)
    save(args.output / "before.json", dict(physical_gpu=4, gpu_uuid=m["gpu_uuid"],
         time=time.time(), gpu4_contexts=cards[4]["contexts"], gpu4_free_mib=cards[4]["free_mib"],
         baseline_identities=baseline, baseline_contexts_all_gpus=baseline_contexts))
    deadline = time.monotonic() + args.timeout
    samples, failed = [], None
    with (args.output / "probe.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                 "--manifest", str(args.manifest.resolve()), "--output", str(args.output.resolve()),
                                 "--child"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            try:
                proc_stat = Path(f"/proc/{proc.pid}/stat").read_text()
            except FileNotFoundError:
                proc_stat = None
                failed = "Probe child exited before identity capture"
            save(args.output / "child.json", dict(pid=proc.pid, start_time=time.time(), proc_stat=proc_stat))
            while proc.poll() is None:
                if failed:
                    break
                current_cards = inventory()[1]
                rows = [dict(gpu=g["index"], uuid=g["uuid"], **p)
                        for g in current_cards for p in g["contexts"] if p["pid"] == proc.pid]
                samples.append(dict(time=time.time(), contexts=rows))
                if any(row["gpu"] != 4 for row in rows):
                    failed = "New probe PID acquired an out-of-scope context"
                    break
                unexpected = {p["pid"] for p in current_cards[4]["contexts"]} - baseline_pids - {proc.pid}
                if unexpected:
                    failed = f"GPU4 acquired an unknown owner during probe: {sorted(unexpected)}"
                    break
                used = sum(int(row["memory"].split()[0]) for row in rows if row["memory"] != "N/A")
                if used > args.memory_limit_mib:
                    failed = "Small camera probe exceeded its VRAM limit"
                    break
                if time.monotonic() > deadline:
                    failed = "Camera probe timed out"
                    break
                time.sleep(0.5)
        finally:
            # Only our still-live child is terminated; no shared Ray/GPU reset.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=15)
        after_cards = inventory()[1]
        leftovers = [dict(gpu=g["index"], uuid=g["uuid"], **p)
                     for g in after_cards for p in g["contexts"] if p["pid"] == proc.pid]
        baseline_after = [dict(gpu=g["index"], uuid=g["uuid"], **p)
                          for g in after_cards for p in g["contexts"] if p["pid"] in baseline_pids]
        baseline_after_identities = []
        for row in baseline:
            try:
                actual = identity(row["pid"])
            except FileNotFoundError:
                actual = None
            baseline_after_identities.append(dict(before=row, after=actual, unchanged=(actual == row)))
        before_keys = {(r["gpu"], r["pid"], r["type"]) for r in baseline_contexts}
        after_keys = {(r["gpu"], r["pid"], r["type"]) for r in baseline_after}
        camera = args.output / "camera.json"
        baseline_unchanged = all(row["unchanged"] for row in baseline_after_identities)
        if not baseline_unchanged and failed is None:
            failed = "Baseline owner identity changed during probe; do not accept this run"
        passed = failed is None and proc.returncode == 0 and camera.is_file() and not leftovers and baseline_unchanged
        report = dict(passed=passed, failure=failed, exit_code=proc.returncode, pid=proc.pid,
                      remaining_contexts=leftovers, samples=samples, time=time.time(),
                      baseline_identities=baseline_after_identities,
                      baseline_contexts_before=baseline_contexts, baseline_contexts_after=baseline_after,
                      baseline_contexts_added=sorted(after_keys - before_keys),
                      baseline_contexts_removed=sorted(before_keys - after_keys),
                      scope="SAPIEN camera plus installed Ray title setter; not an actual RLinf Ray worker")
        save(args.output / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}))
        if not passed:
            raise RuntimeError("GPU scope probe failed; inspect its report/log before any RLT/native launch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--memory-limit-mib", type=int, default=4096)
    parser.add_argument("--allow-existing-owned-pids", type=Path,
                        help="Explicit JSON [{pid,uid:20001,start}] for a shared-GPU4 probe; requires 16 GiB free")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    (child if args.child else parent)(args)


if __name__ == "__main__":
    main()
