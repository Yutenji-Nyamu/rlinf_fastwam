"""Reuse native Dojo lifecycle inside an already registered outer GPU5 owner."""

import os
import sys
from pathlib import Path


def main(directory):
    assert (
        os.environ.get("APPENDIX_STAGE_ID")
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "5"
    )
    from tools.dsrl_accel_v2 import dojo_owner

    d = Path(directory)
    p = dojo_owner.read(d / "plan.json")
    assert p["gpu"] == 5 and p["pre_registered"] and not p["predecessors"]
    # This is a local child handshake only; the parent retains the global reservation.
    dojo_owner.save(d / "registered.json", dict(identity=dojo_owner.proc(os.getpid())))
    dojo_owner.main(d)


if __name__ == "__main__":
    main(sys.argv[1])
