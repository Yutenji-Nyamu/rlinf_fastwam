"""Run a prepared, bounded repair batch before the later value-analysis stage."""

import json
import os
import subprocess
from pathlib import Path

from .common import write_json
from .pipeline import cleanup


def run(data, plan_name="pending-repairs.json"):
    path = Path(data) / plan_name
    if not path.exists():
        return
    plan = json.loads(path.read_text())
    out = Path(data) / (
        "repairs" if plan_name == "pending-repairs.json" else "dojo-repairs"
    )
    out.mkdir(exist_ok=True)
    tag = os.environ["APPENDIX_STAGE_ID"]
    results = []
    for job in plan["jobs"]:
        assert job["module"] in {
            "rt_sampling",
            "rt_appearance",
            "mcr_features",
            "native_stage",
        }
        if not all(Path(p).exists() for p in job.get("requires", [])):
            results.append(dict(name=job["module"], state="skipped_missing_input"))
            continue
        env = dict(os.environ, **job["environment"])
        env.update(APPENDIX_STAGE_ID=tag, CUDA_VISIBLE_DEVICES="5")
        command = [
            job["python"],
            "-u",
            "-B",
            "-m",
            "tools.appendix_signals." + job["module"],
            *job["args"],
        ]
        with (out / (job.get("name", job["module"]) + ".log")).open("xb") as log:
            try:
                code = subprocess.run(
                    command,
                    cwd=job["cwd"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=job["timeout"],
                ).returncode
            except subprocess.TimeoutExpired:
                code = 124
            finally:
                cleanup(tag)
        results.append(
            dict(
                name=job["module"],
                returncode=code,
                state="complete" if code == 0 else "failed_or_bounded",
            )
        )
        write_json(out / "results.json", results)
    path.rename(path.with_name(path.stem + "-submitted.json"))
