# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Accepted pair launcher: MFD 12 updates -> resume to 14 -> N1 export -> STOP.

Usage: python mfd_smoke.py DEPLOYMENT_DIR
The prepared deployment provides physical_gpus and existing per-card environment
files. No formal training, evaluator, scheduler or fallback is started here.
"""

import os
import traceback

import run_queue as q


def main():
    q.save(q.D / "smoke-identity.json", q.proc(os.getpid()))
    q.event("MFD_SMOKE_12")
    flags = ["--decode-workers", "8", "--prefetch-batches", "1", "--profile-stages"]
    output = q.O / "smoke-mfd"
    first = q.train("mfd", output, 12, label="mfd-12", extra_args=flags)
    q.event("MFD_RESUME_14")
    q.train("mfd", output, 14, first["path"], label="mfd-resume-14", extra_args=flags)
    q.accept_smoke("mfd")
    receipt = q.read(q.D / "smoke-mfd-accepted.json")
    for check in receipt["checks"]:
        row = check["last_update"]
        assert row["student_updates"] == 14 and row["auxiliary_updates"] == 70
        assert row["mfd_guidance_nfe"] >= 2
    q.event("MFD_SMOKE_COMPLETE", formal_started=False)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        q.event("MFD_SMOKE_FAILED", error=traceback.format_exc())
        raise
