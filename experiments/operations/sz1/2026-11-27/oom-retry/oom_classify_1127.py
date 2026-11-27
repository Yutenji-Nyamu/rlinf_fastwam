"""Find explicit device-memory errors in one attempt's stdout/stderr lines."""

from collections import deque
from itertools import groupby
import re


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CUDA = re.compile(
    r"(?:^\s*|\b(?:RuntimeError|OutOfMemoryError|AcceleratorError):\s*)"
    r"CUDA(?: error:)?\s+out of memory\b|\bCUDA error:\s*out of memory\b",
    re.IGNORECASE,
)
_STATUS = re.compile(
    r"(?:^\s*|\b(?:error:|returned|failed with)\s+)"
    r"(?P<code>VK_ERROR_OUT_OF_DEVICE_MEMORY|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation)\b",
    re.IGNORECASE,
)
_OIDN = re.compile(
    r"\[svulkan2\]\s*\[error\]\s*OIDN Error:\s*out of memory[.!]?\s*$",
    re.IGNORECASE,
)


def find_gpu_oom(lines):
    """Return first explicit error and final 8 lines, or None; pass only this attempt.

    OIDN is reported as a renderer allocation error, distinct from a CUDA
    allocator traceback. This function neither reads files nor changes a process.
    """
    tail = deque(maxlen=8)
    evidence = None
    for number, raw in enumerate(lines, 1):
        line = _ANSI.sub("", raw).rstrip("\r\n")
        tail.append(line[-1200:])
        if evidence is not None:
            continue
        if _CUDA.search(line):
            kind = "cuda"
        elif _OIDN.search(line):
            kind = "oidn_renderer"
        else:
            status = _STATUS.search(line)
            if status is None:
                continue
            kind = "vulkan" if status["code"].upper().startswith("VK_") else "cuda"
        evidence = {"kind": kind, "line_number": number, "line": line}
    if evidence is not None:
        evidence["tail"] = list(tail)
    return evidence


RPC_FAILURE = re.compile(r"SimulatorTransportError|EOFError|BrokenPipeError|Connection(?:Reset|Refused|Aborted)Error|"
                         r"(?:RPC|socket|native server).{0,80}(?:timed out|timeout|disconnected|closed|reset|EOF)|"
                         r"Connection reset by peer", re.IGNORECASE)
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
                    if not rpc and RPC_FAILURE.search(line):
                        rpc = line.strip()[-800:]
                    if not transport and any(token in line for token in TRANSPORT_ERRORS):
                        transport = True
                    yield line
            oom = find_gpu_oom(observed())
    return step, oom, rpc, transport
