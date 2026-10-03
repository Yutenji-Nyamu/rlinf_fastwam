"""Early, fail-closed graphics visibility for explicitly marked EXPO interpreters."""
import os

_manifest_path = os.environ.get("RLINF_EXPO_GPU_SCOPE_MANIFEST")
if _manifest_path:
    try:
        import ctypes
        import hashlib
        import json
        import socket
        import time
        from pathlib import Path

        _m = json.loads(Path(_manifest_path).read_text())
        if os.getuid() != _m["uid"] or socket.gethostname() != _m["hostname"]:
            raise RuntimeError("EXPO GPU scope host/uid mismatch")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(_m["gpu_uuids"]):
            raise RuntimeError("EXPO CUDA UUID order differs from scope")
        _profile = Path(_m["profile_path"])
        if _profile.stat().st_uid != os.getuid() or hashlib.sha256(_profile.read_bytes()).hexdigest() != _m["profile_sha256"]:
            raise RuntimeError("EXPO graphics profile identity/hash mismatch")
        _name = _m["commname"].encode("ascii")
        if not 0 < len(_name) <= 15:
            raise RuntimeError("Invalid EXPO process commname")
        _libc = ctypes.CDLL(None, use_errno=True)
        if _libc.prctl(15, _name, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "EXPO prctl failed")
        if Path("/proc/self/comm").read_text().strip() != _m["commname"]:
            raise RuntimeError("EXPO process name verification failed")
        os.environ["__GL_APPLICATION_PROFILE"] = "1"
        _receipt = Path(_m["receipt_directory"]) / (str(os.getpid()) + "-" + str(time.time_ns()) + ".json")
        with _receipt.open("x") as _f:
            json.dump({"pid": os.getpid(), "ppid": os.getppid(), "time": time.time(), "commname": _m["commname"], "gpu_uuids": _m["gpu_uuids"], "profile_sha256": _m["profile_sha256"]}, _f)
    except BaseException as _exc:
        os.write(2, ("EXPO GPU SCOPE FAILED: " + str(_exc) + "\n").encode())
        os._exit(78)
