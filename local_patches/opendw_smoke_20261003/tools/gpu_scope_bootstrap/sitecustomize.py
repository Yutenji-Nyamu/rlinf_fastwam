"""Fail before GPU initialization if an explicitly requested job scope is invalid."""
import os

_manifest = os.environ.get("RLINF_OPENDW_GPU_SCOPE_MANIFEST")
if _manifest:
    try:
        from gpu_scope_runtime import install
        install(_manifest)
    except BaseException as _error:
        os.write(2, ("OPENDW GPU SCOPE FAILED: " + str(_error) + "\n").encode())
        os._exit(78)
