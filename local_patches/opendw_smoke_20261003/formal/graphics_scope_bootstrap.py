"""Copied as private sitecustomize.py; never installed in global site-packages."""
import os

_manifest = os.environ.get("RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST")
if _manifest:
    try:
        from graphics_scope_runtime import install
        install(_manifest)
    except BaseException as _error:
        os.write(2, ("OPENDW FORMAL GRAPHICS SCOPE FAILED: " + str(_error) + "\n").encode())
        os._exit(78)
