"""Recognize GPU allocation errors and the native RoboTwin camera-buffer traceback."""

from collections import deque
from itertools import groupby
import re


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CUDA = re.compile(
    r"(?:^\s*|\b(?:RuntimeError|OutOfMemoryError|AcceleratorError):\s*)"
    r"CUDA(?: error:)?\s+out of memory\b|\bCUDA error:\s*out of memory\b",
    re.IGNORECASE,
)
_NAMED = re.compile(
    r"\btorch\.cuda\.OutOfMemoryError\s*:|"
    r"\b(?:vk::OutOfDeviceMemoryError|ErrorOutOfDeviceMemory)\b|"
    r"\b(?:CUDA|Vulkan)\s+(?:error\s*[:(]?\s*2\)?\s*[:,-]\s*)?out of (?:device )?memory\b",
    re.IGNORECASE,
)
_STATUS = re.compile(
    r"(?:^\s*|\b(?:error:|returned|failed with|failed:|result:)\s+)"
    r"(?P<code>VK_ERROR_OUT_OF_DEVICE_MEMORY|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation)\b",
    re.IGNORECASE,
)
_OIDN = re.compile(
    r"\[svulkan2\]\s*\[error\]\s*OIDN Error:\s*out of memory[.!]?\s*$",
    re.IGNORECASE,
)


_CAMERA_FRAME = re.compile(r'/envs/camera/camera\.py", line \d+, in update_picture\s*$')
_CAMERA_CALL = re.compile(r'\bself\.\w+\.take_picture\(\)\s*$')
_ENV_WORKER = re.compile(r'ray::EnvWorker\.\w+\(\).*?pid=(\d+).*?EnvGroup\(rank=(\d+)\)')
_NON_DEVICE_MEMORY = re.compile(r'VK_ERROR_(?:DEVICE_LOST|OUT_OF_HOST_MEMORY)|vk::(?:DeviceLost|OutOfHostMemory)Error', re.I)


def find_gpu_oom(lines):
    """Return first explicit error and final 8 lines, or None; pass only this attempt.

    OIDN is reported as a renderer allocation error, distinct from a CUDA
    allocator traceback. This function neither reads files nor changes a process.
    """
    tail = deque(maxlen=8)
    evidence = None
    env_worker = None
    conflicting_error = False
    for number, raw in enumerate(lines, 1):
        line = _ANSI.sub("", raw).rstrip("\r\n")
        tail.append(line[-1200:])
        if evidence is not None and evidence["kind"] != "robotwin_camera_buffer":
            continue
        if "ray::EnvWorker." in line:
            match = _ENV_WORKER.search(line)
            env_worker = (number, int(match[1]), int(match[2])) if match else None
        if _NON_DEVICE_MEMORY.search(line):
            conflicting_error = True
        if _CUDA.search(line):
            kind = "cuda"
        elif _NAMED.search(line) and not any(t in line.lower() for t in ("enable checks","out of host memory","defaultcpuallocator")):
            kind = "vulkan" if any(t in line.lower() for t in ("vk::", "vulkan", "outofdevicememory")) else "cuda"
        elif (line.endswith("RuntimeError: cannot create buffer") and len(tail) >= 3
              and _CAMERA_FRAME.search(tail[-3]) and _CAMERA_CALL.search(tail[-2])):
            evidence = dict(kind="robotwin_camera_buffer", line_number=number, line=line)
            if env_worker and number - env_worker[0] <= 24:
                evidence.update(worker_pid=env_worker[1], env_rank=env_worker[2])
            continue
        elif ("[omni.physx.plugin]" in line and
              "PhysX ABORT error: PhysX failed to allocate GPU memory - aborting simulation." in line):
            kind = "physx_gpu_oom"
        elif _OIDN.search(line):
            kind = "oidn_renderer"
        else:
            status = _STATUS.search(line)
            if status is None:
                continue
            kind = "vulkan" if status["code"].upper().startswith("VK_") else "cuda"
        evidence = {"kind": kind, "line_number": number, "line": line}
    if conflicting_error and evidence and evidence["kind"] == "robotwin_camera_buffer":
        return None
    if evidence is not None:
        evidence["tail"] = list(tail)
    return evidence


RPC_FAILURE = re.compile(r"SimulatorTransportError|EOFError|BrokenPipeError|Connection(?:Reset|Refused|Aborted)Error|"
                         r"(?:RPC|socket|native server).{0,80}(?:timed out|timeout|disconnected|closed|reset|EOF)|"
                         r"TimeoutError:\s*Dojo step timed out on GPU \d+\b|Connection reset by peer", re.IGNORECASE)
STEP = re.compile(r"│\s*Global Step:\s*(\d+)/\d+\s*│\s*Progress:")
TRANSPORT_ERRORS = ("SimulatorTransportError:", "Fatal Python error: PyGILState_Release",
                    "RoboTwin simulator RPC exceeded")
UNRELATED = re.compile(r"(?:^|\s)(?:[\w.]+\.)?(?:KeyError|FileNotFoundError|ValueError|"
                       r"AssertionError|TypeError|PermissionError|ImportError|ModuleNotFoundError):")


def driver_failure(lines):
    """Read all of the final unfinished round, including Ray's nested exceptions."""
    step, oom, rpc, transport = None, None, None, False
    def boundary(line):
        if STEP.search(line):
            return "progress"
        return "other_error" if UNRELATED.search(line) else None
    for event, segment in groupby(lines, key=boundary):
        if event == "progress":
            for line in segment:
                step = int(STEP.search(line)[1])
            oom, rpc, transport = None, None, False
        elif event == "other_error":
            oom, rpc, transport = None, None, False
        else:
            def observed():
                nonlocal rpc, transport
                for line in segment:
                    is_dojo_timeout = re.search(r"TimeoutError:\s*Dojo step timed out on GPU \d+\b", line)
                    is_config = re.match(r"^\s*['\"]?(?:rpc|socket|native_server)_timeout['\"]?\s*[:=]\s*[0-9]", line, re.I)
                    if is_dojo_timeout or (not rpc and not is_config and RPC_FAILURE.search(line)):
                        rpc = line.strip()[-800:]
                    if not transport and any(token in line for token in TRANSPORT_ERRORS):
                        transport = True
                    yield line
            oom = find_gpu_oom(observed())
    return step, oom, rpc, transport
