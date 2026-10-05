"""Opt-in, per-worker GPU4-7 graphics scope, derived from the GPU4 v5 runtime.

The worker's single-device CVD selects a physical card before GPU imports. The
one observed full-node CPU ChannelWorker mask is narrowed to GPU4; CPU discovery
with an unset/empty mask is unchanged. Rendering always requires a single card.
"""
import hashlib
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import pwd
import socket
import stat
import sys
import time

MANIFEST_ENV = "RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST"
LEGACY_MANIFEST_ENV = "RLINF_OPENDW_GPU_SCOPE_MANIFEST"
PHYSICAL_GPUS = (4, 5, 6, 7)
FULL_NODE_MASK = "0,1,2,3,4,5,6,7"


def select_target(manifest, cuda_visibility):
    """Pure selector; no CUDA import or environment mutation."""
    if manifest.get("physical_gpus") != list(PHYSICAL_GPUS):
        raise RuntimeError("Formal scope requires the explicit physical GPU4-7 inventory")
    cards = manifest.get("cards", {})
    if set(cards) != {str(gpu) for gpu in PHYSICAL_GPUS}:
        raise RuntimeError("Formal scope needs exactly four pinned card identities")
    if cuda_visibility in (None, "", FULL_NODE_MASK):
        return 4
    for gpu in PHYSICAL_GPUS:
        if cuda_visibility in (str(gpu), cards[str(gpu)]["gpu_uuid"]):
            return gpu
    raise RuntimeError(f"Worker CUDA visibility is not a single authorized card: {cuda_visibility!r}")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned_file(path, expected_sha=None):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise RuntimeError(f"Scope file must be an owned non-writable regular file: {path}")
    if expected_sha is not None and digest(path) != expected_sha:
        raise RuntimeError(f"Scope file changed: {path}")
    return path


def read_manifest(path):
    m = json.loads(owned_file(path).read_text())
    if m["uid"] != os.getuid() or m["hostname"] != socket.gethostname():
        raise RuntimeError("Scope host/UID mismatch")
    target = select_target(m, os.environ.get("CUDA_VISIBLE_DEVICES"))
    if m.get("schema") != 1 or m.get("cpu_full_mask_target") != 4:
        raise RuntimeError("Unrecognized formal graphics contract")
    if m.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise RuntimeError("Scope preparation predates the current host boot")
    for gpu in PHYSICAL_GPUS:
        card = m["cards"][str(gpu)]
        if card["physical_gpu"] != gpu or not card["gpu_uuid"].startswith("GPU-"):
            raise RuntimeError("Invalid pinned physical card identity")
        if not 0 <= card["minor"] < 32 or card["mask"] != 1 << card["minor"]:
            raise RuntimeError("Invalid physical minor/mask contract")
    if len({row["gpu_uuid"] for row in m["cards"].values()}) != 4:
        raise RuntimeError("Duplicate pinned GPU UUID")
    if len({row["commname"] for row in m["cards"].values()}) != 4:
        raise RuntimeError("Duplicate per-card process commname")
    for key in ("marker", "profile", "bootstrap", "runtime"):
        owned_file(m[key + "_path"], m[key + "_sha256"])
    for existing, record in m["existing_profiles"].items():
        if digest(existing) != record["sha256"]:
            raise RuntimeError(f"NVIDIA profile changed since read-only audit: {existing}")
    for name in m["profile_search_paths"]:
        root = Path(name)
        if not root.exists():
            continue
        paths = root.iterdir() if root.is_dir() else [root]
        for path in paths:
            if not path.is_file() or str(path) == m["profile_path"] or str(path) in m["existing_profiles"]:
                continue
            if path.stat().st_size > 4 * 1024 * 1024 or b"EGLVisibleDGPUDevices" in path.read_bytes():
                raise RuntimeError(f"New NVIDIA device rule requires precedence audit: {path}")
    # Management-tidy gave the existing inline settings explicit names. Accept
    # only that representation change; rule order, patterns and masks stay exact.
    actual_profile = json.loads(Path(m["profile_path"]).read_text())
    if not isinstance(actual_profile, dict) or set(actual_profile) != {"rules"}:
        raise RuntimeError("Graphics profile must contain exactly the four explicit rules")
    rules = actual_profile["rules"]
    if not isinstance(rules, list) or len(rules) != len(PHYSICAL_GPUS):
        raise RuntimeError("Graphics profile must contain exactly the four explicit rules")
    for gpu, rule in zip(PHYSICAL_GPUS, rules):
        if not isinstance(rule, dict) or set(rule) != {"pattern", "profile"}:
            raise RuntimeError("Graphics rule contains unexpected fields")
        profile = rule["profile"]
        if isinstance(profile, dict):
            expected_name = "chenyiteng-" + m["cards"][str(gpu)]["commname"]
            if set(profile) != {"name", "settings"} or profile["name"] != expected_name:
                raise RuntimeError("Named graphics profile differs from the pinned commname")
            rule["profile"] = profile["settings"]
    expected_profile = {"rules": [
        {"pattern": {"feature": "commname", "matches": m["cards"][str(gpu)]["commname"]},
         "profile": ["EGLVisibleDGPUDevices", m["cards"][str(gpu)]["mask"]]}
        for gpu in PHYSICAL_GPUS
    ]}
    if actual_profile != expected_profile:
        raise RuntimeError("Graphics profile does not match the four explicit card rules")
    m.update(m["cards"][str(target)])
    return m


def check_marker(m):
    marker = str(Path(m["marker_path"]).resolve(strict=True))
    mapped = {line.split(maxsplit=5)[5] for line in Path("/proc/self/maps").read_text().splitlines()
              if len(line.split(maxsplit=5)) == 6}
    if marker not in mapped:
        raise RuntimeError("Scope marker was not preloaded before Python/GPU libraries")


def pin_main_comm(m):
    name = m.get("commname")
    if name is None:  # Read-only diagnostics may inspect historical v1 manifests.
        return
    encoded = name.encode("ascii")
    if not 1 <= len(encoded) <= 15 or "\n" in name or "\0" in name:
        raise RuntimeError("Invalid private process commname")
    # prctl(PR_SET_NAME) changes only the calling thread. Ray can import an
    # actor's environment from a non-main thread, so target the group leader.
    target = Path(f"/proc/self/task/{os.getpid()}/comm")
    target.write_text(name)
    if Path("/proc/self/comm").read_text().strip() != name:
        raise RuntimeError("Failed to pin the main thread commname")


def preserve_ray_titles(m):
    # The installed Ray calls this extension entry point from worker.connect
    # and _changeproctitle. Importing Ray does not connect or create a job.
    import ray
    original = ray._raylet.setproctitle
    if getattr(original, "_opendw_formal_scope_token", None) == m["token"]:
        return True
    if (getattr(original, "_opendw_scope_token", None) is not None or
            getattr(original, "_opendw_formal_scope_token", None) is not None):
        raise RuntimeError("Ray process title setter already belongs to another scope")

    @functools.wraps(original)
    def scoped_title(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            # Preserve Ray's argv/process title; only restore /proc/PID/comm.
            pin_main_comm(m)

    scoped_title._opendw_formal_scope_token = m["token"]
    ray._raylet.setproctitle = scoped_title
    pin_main_comm(m)
    return True


def verify_ray_title_scope(m):
    """CPU-only stress of the installed Ray setter, without ray.init()."""
    import ray
    import threading
    if getattr(ray._raylet.setproctitle, "_opendw_formal_scope_token", None) != m["token"]:
        raise RuntimeError("The installed Ray title entry point is not scoped")
    ray._raylet.setproctitle("ray::EnvWorker.scope_probe")
    if Path("/proc/self/comm").read_text().strip() != m["commname"]:
        raise RuntimeError("Ray main-thread title change escaped private commname")
    errors = []

    def change_from_thread():
        try:
            ray._raylet.setproctitle("ray::EnvWorker.scope_probe_thread")
        except BaseException as exc:
            errors.append(repr(exc))

    thread = threading.Thread(target=change_from_thread)
    thread.start()
    thread.join(timeout=5)
    if thread.is_alive() or errors or Path("/proc/self/comm").read_text().strip() != m["commname"]:
        raise RuntimeError(f"Ray background-thread title change escaped scope: {errors}")
    if b"ray::EnvWorker.scope_probe_thread" not in Path("/proc/self/cmdline").read_bytes():
        raise RuntimeError("Ray argv title was not preserved")
    if ray.is_initialized():
        raise RuntimeError("A CPU title probe must not connect to Ray")
    return dict(api="ray._raylet.setproctitle", main_thread=True, background_thread=True,
                argv_preserved=True, ray_initialized=False)


def check_cuda(m, *, require=False):
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    # NodePlacementStrategy gives CPU ChannelWorkers the explicit full node
    # inventory. Narrow only this exact observed mask before Ray/Torch imports;
    # discovery drivers with no mask and explicit wrong-device masks stay as-is.
    if value == FULL_NODE_MASK:
        if m["physical_gpu"] != 4:
            raise RuntimeError("CPU full-node compatibility is only permitted on GPU4")
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        value = "4"
    # RLinf's physical placement sets a single physical card before exec. Do not mask its CPU
    # discovery driver/managers: they need the physical cluster inventory.
    if value in (None, ""):
        if require:
            raise RuntimeError("Native rendering requires explicit single-card CUDA visibility")
        return value
    if value not in (str(m["physical_gpu"]), m["gpu_uuid"]):
        raise RuntimeError(f"CUDA visibility differs from pinned GPU{m['physical_gpu']}: {value!r}")
    # RLinf's accelerator parser requires numeric CVD. Never rewrite its number
    # to a UUID; bind_renderer verifies the actual logical CUDA0 identity.
    return value


def verify_cuda_mapping(m):
    """Check the one visible CUDA device without creating a CUDA context."""
    import ctypes as C
    check_cuda(m, require=True)
    cuda = C.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [C.c_uint]
    cuda.cuDeviceGetCount.argtypes = [C.POINTER(C.c_int)]
    cuda.cuDeviceGet.argtypes = [C.POINTER(C.c_int), C.c_int]
    cuda.cuDeviceGetPCIBusId.argtypes = [C.c_char_p, C.c_int, C.c_int]
    uuid_getter = getattr(cuda, "cuDeviceGetUuid_v2", cuda.cuDeviceGetUuid)
    uuid_getter.argtypes = [C.c_void_p, C.c_int]
    count, device, pci, uuid = C.c_int(), C.c_int(), C.create_string_buffer(32), (C.c_ubyte * 16)()
    if not (cuda.cuInit(0) == 0 and cuda.cuDeviceGetCount(C.byref(count)) == 0
            and count.value == 1 and cuda.cuDeviceGet(C.byref(device), 0) == 0
            and cuda.cuDeviceGetPCIBusId(pci, len(pci), device) == 0
            and uuid_getter(C.byref(uuid), device) == 0):
        raise RuntimeError("Cannot identify the one visible CUDA device")
    raw = bytes(uuid).hex()
    actual_uuid = "GPU-" + "-".join((raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]))
    actual_pci = pci.value.decode()
    if actual_uuid.lower() != m["gpu_uuid"].lower() or actual_pci.lower()[-10:] != m["pci"].lower()[-10:]:
        raise RuntimeError(f"Logical CUDA0 is outside GPU{m['physical_gpu']} scope: {actual_uuid}, {actual_pci}")
    return dict(cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                cuda_device_count=count.value, cuda_pci=actual_pci, cuda_uuid=actual_uuid)


def write_receipt(m, phase, **fields):
    start_ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    data = dict(phase=phase, pid=os.getpid(), ppid=os.getppid(), time=time.time(),
                start_ticks=start_ticks,
                token=m["token"], physical_gpu=m["physical_gpu"], gpu_uuid=m["gpu_uuid"],
                comm=Path("/proc/self/comm").read_text().strip(), **fields)
    target = Path(m["receipts_dir"]) / f"{os.getpid()}-{time.time_ns()}-{phase}.json"
    with target.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    return data


def bind_renderer(m):
    """Same SAPIEN 3.0.1 Scene binding as verified EXPO, restricted to this process."""
    check_marker(m)
    check_cuda(m, require=True)
    pin_main_comm(m)
    mapping = verify_cuda_mapping(m)
    import sapien
    from sapien.wrapper.scene import Scene
    if getattr(Scene, "_opendw_gpu_scope", None):
        raise RuntimeError("Do not combine a legacy graphics runtime with the formal runtime")
    existing = getattr(Scene, "_opendw_formal_graphics_scope", None)
    if existing:
        if existing["gpu_uuid"] != m["gpu_uuid"] or existing["token"] != m["token"]:
            raise RuntimeError("SAPIEN already bound to another scope")
        return existing
    if str(sapien.__version__) != "3.0.1" or sapien.Scene is not Scene:
        raise RuntimeError("Native scope requires the audited SAPIEN 3.0.1 Scene API")
    original = Scene.__init__
    expected = (
        "def __init__(self, systems=None):\n"
        "        if systems is None:\n"
        "            super().__init__(\n"
        "                [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()]\n"
        "            )\n"
        "        else:\n"
        "            super().__init__(systems)"
    )
    if inspect.getsource(original).strip() != expected:
        raise RuntimeError("SAPIEN Scene initializer differs from audited source")

    def scoped_init(self, systems=None):
        pin_main_comm(m)
        if systems is None:
            systems = [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem("cuda:0")]
        original(self, systems)

    Scene.__init__ = scoped_init
    receipt = write_receipt(m, "renderer_bound", api="Scene.RenderSystem(cuda:0)", **mapping)
    Scene._opendw_formal_graphics_scope = receipt
    return receipt


class _NativeLoader(importlib.abc.Loader):
    def __init__(self, delegate, m):
        self.delegate, self.m = delegate, m

    def create_module(self, spec):
        method = getattr(self.delegate, "create_module", None)
        return method(spec) if method else None

    def exec_module(self, module):
        # Runs before VectorEnv's imports can create RoboTwin scenes. It does
        # not eagerly import SAPIEN in actor, rollout, or management workers.
        bind_renderer(self.m)
        self.delegate.exec_module(module)


class _NativeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, m):
        self.m = m

    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("torch", "sapien"):
            pin_main_comm(self.m)
            write_receipt(self.m, "before_" + fullname)
            return None
        if fullname != "robotwin.envs.vector_env":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot locate native RoboTwin VectorEnv")
        spec.loader = _NativeLoader(spec.loader, self.m)
        return spec


def install(path):
    if os.environ.get(LEGACY_MANIFEST_ENV):
        raise RuntimeError("Legacy GPU4 scope and formal graphics scope cannot share an interpreter")
    m = read_manifest(path)
    check_marker(m)
    if os.environ.get("__GL_APPLICATION_PROFILE") != "1":
        raise RuntimeError("Application-profile support must be enabled before process launch")
    if os.environ.get("HOME") != m.get("home") or m.get("home") != pwd.getpwuid(os.getuid()).pw_dir:
        raise RuntimeError("NVIDIA profile HOME differs from the manifest and owner account")
    check_cuda(m)
    pin_main_comm(m)
    titles_patched = preserve_ray_titles(m) if m.get("commname") else False
    if "robotwin.envs.vector_env" in sys.modules:
        raise RuntimeError("GPU scope bootstrap loaded after native environment")
    sys.meta_path.insert(0, _NativeFinder(m))
    write_receipt(m, "bootstrap", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  setproctitle_patched=titles_patched,
                  setproctitle_module=getattr(sys.modules.get("ray._raylet"), "__file__", None),
                  setproctitle_api="ray._raylet.setproctitle", home=os.environ.get("HOME"))

