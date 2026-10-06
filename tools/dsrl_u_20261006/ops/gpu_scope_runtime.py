"""Process-local the assigned GPU graphics scope. Imported only by an explicit private bootstrap."""
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import socket
import stat
import sys
import time

MANIFEST_ENV = "RLINF_OPENDW_GPU_SCOPE_MANIFEST"


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
    if m["physical_gpu"] not in (4, 5, 6, 7) or not m["gpu_uuid"].startswith("GPU-"):
        raise RuntimeError("This scope only permits physical the assigned GPU")
    if not 0 <= m["minor"] < 32 or m["mask"] != 1 << m["minor"]:
        raise RuntimeError("Invalid physical minor/mask contract")
    for key in ("marker", "profile", "bootstrap", "runtime"):
        owned_file(m[key + "_path"], m[key + "_sha256"])
    return m


def check_marker(m):
    marker = str(Path(m["marker_path"]).resolve(strict=True))
    mapped = {line.split(maxsplit=5)[5] for line in Path("/proc/self/maps").read_text().splitlines()
              if len(line.split(maxsplit=5)) == 6}
    if marker not in mapped:
        raise RuntimeError("Scope marker was not preloaded before Python/GPU libraries")


def check_cuda(m, *, require=False):
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    # RLinf's physical placement sets '4' before exec. Do not mask its CPU
    # discovery driver/managers: they need the physical cluster inventory.
    if value in (None, ""):
        if require:
            raise RuntimeError("Native rendering requires explicit the assigned GPU CUDA visibility")
        return value
    if value not in (str(m["physical_gpu"]), m["gpu_uuid"]):
        raise RuntimeError(f"CUDA visibility exceeds the assigned GPU scope: {value!r}")
    return value  # RLinf requires physical integer IDs; preserve its placement.


def write_receipt(m, phase, **fields):
    data = dict(phase=phase, pid=os.getpid(), ppid=os.getppid(), time=time.time(),
                token=m["token"], physical_gpu=m["physical_gpu"], gpu_uuid=m["gpu_uuid"],
                comm=Path("/proc/self/comm").read_text().strip(), **fields)
    target = Path(m["receipts_dir"]) / f"{os.getpid()}-{time.time_ns()}-{phase}.json"
    with target.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    return data


def set_scope_comm(gpu):
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(15, ctypes.c_char_p(("dsrl-u-g" + str(gpu)).encode()), 0, 0, 0):
        raise OSError(ctypes.get_errno(), "Cannot apply the existing the assigned GPU EGL profile")


def bind_renderer(m):
    """Same SAPIEN 3.0.1 Scene binding as verified EXPO, restricted to this process."""
    check_marker(m)
    check_cuda(m, require=True)
    set_scope_comm(m["physical_gpu"])
    import sapien
    from sapien.wrapper.scene import Scene
    existing = getattr(Scene, "_opendw_gpu_scope", None)
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
        if systems is None:
            systems = [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem("cuda:0")]
        original(self, systems)

    Scene.__init__ = scoped_init
    receipt = write_receipt(m, "renderer_bound", api="Scene.RenderSystem(cuda:0)")
    Scene._opendw_gpu_scope = receipt
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
        if fullname != "robotwin.envs.vector_env":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot locate native RoboTwin VectorEnv")
        spec.loader = _NativeLoader(spec.loader, self.m)
        return spec


def install(path):
    m = read_manifest(path)
    set_scope_comm(m["physical_gpu"])
    check_marker(m)
    if os.environ.get("__GL_APPLICATION_PROFILE") != "1":
        raise RuntimeError("Application-profile support must be enabled before process launch")
    # CPU Ray channel/manager workers may expose the cluster inventory.
    # Validate CUDA only when a native GPU renderer is actually constructed.
    if "robotwin.envs.vector_env" in sys.modules:
        raise RuntimeError("GPU scope bootstrap loaded after native environment")
    sys.meta_path.insert(0, _NativeFinder(m))
    write_receipt(m, "bootstrap", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))

