"""Opt-in, environment-local loading of the pinned scene-access fence fix."""

import ctypes
import os
import sys
import sysconfig
import threading
from pathlib import Path

_handles = None
_lock = threading.Lock()


def load_scene_fence():
    """Load before SAPIEN, after PyTorch, without exporting dependency symbols."""
    global _handles
    library = os.environ.get("RLINF_SCENE_FENCE_LIBRARY")
    if not library:
        return None
    with _lock:
        if _handles is not None:
            return _handles[-1]
        if "sapien.pysapien" in sys.modules:
            raise RuntimeError("Scene-fence fix must be loaded before SAPIEN")
        if "torch" not in sys.modules:
            raise RuntimeError("Scene-fence fix must be loaded after PyTorch")
        site = Path(sysconfig.get_path("purelib"))
        oidn = site / "sapien" / "oidn_library"
        # Same pinned files/order as SAPIEN's _oidn_tricks.py. No device creation.
        core = ctypes.CDLL(str(oidn / "libOpenImageDenoise_core.so.2.0.1"), mode=ctypes.RTLD_LOCAL)
        api = ctypes.CDLL(str(oidn / "libOpenImageDenoise.so.2.0.1"), mode=ctypes.RTLD_LOCAL)
        patch = ctypes.CDLL(str(Path(library).resolve()), mode=os.RTLD_NOW | ctypes.RTLD_LOCAL)
        _handles = (core, api, patch)
        return patch
