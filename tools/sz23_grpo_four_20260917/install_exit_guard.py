"""Install one frozen, one-shot exit companion for each of this host's two runs."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ST = Path("/data/chenyiteng/deployment-20260917/grpo-four")
spec = importlib.util.spec_from_file_location("sz23_exit_guard", ST / "exit_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def install(run_spec):
    runtime = Path(run_spec["run"]) / "runtime"
    contract = guard.load(runtime / "contract.json")
    guard.require(contract["namespace"] == run_spec["namespace"] and contract["run"] == run_spec["run"], "Plan/runtime identity mismatch")
    head = guard.source_check(runtime, contract)
    driver = guard.load(runtime / "driver-identity.json")
    guard.require(driver["uid"] == 20001 and driver["namespace"] == contract["namespace"] and guard.same(driver), "Original driver is not alive with the frozen identity")
    wrapper = guard.load(runtime / "wrapper-identity.json")
    guard.require(wrapper["uid"] == 20001 and guard.same(wrapper), "Original wrapper identity is not alive")
    snapshot = guard.snapshot_subprocess(runtime, contract)
    rows = snapshot["actors"]
    guard.require(len(snapshot["nodes"]) == 1 and snapshot["nodes"][0]["gpus"] == 8, "Unexpected personal Ray cluster")
    guard.require(len(rows) == 15 and all(row["state"] == "ALIVE" and row["ray_namespace"] == contract["namespace"] for row in rows), "Expected all 15 actors in this namespace")
    jobs = {row["job_id"] for row in rows}
    guard.require(len(jobs) == 1, "Namespace does not have one training job")
    job_id = next(iter(jobs))
    identities = [guard.proc(row["pid"]) for row in rows]
    guard.require(all(identity and identity["uid"] == 20001 and identity["state"] != "Z" for identity in identities), "Training actor UID/liveness mismatch")
    guard.require(guard.same(driver), "Original driver exited during installation checks")
    directory = runtime / "exit-guard-20260917"
    guard.require(not directory.exists(), "Exit guard installation already attempted")
    directory.mkdir(exist_ok=False)
    script_hash = guard.sha(ST / "exit_guard.py")
    frozen = {"schema_version": 1, "time": guard.now(), "uid": 20001, "hostname": socket.gethostname(),
              "run": contract["run"], "namespace": contract["namespace"], "ray_address": contract["ray_address"],
              "source_head": head, "runtime_contract_sha256": guard.sha(runtime / "contract.json"),
              "guard_script_sha256": script_hash, "guard_script": str(ST / "exit_guard.py"),
              "driver": driver, "wrapper": wrapper, "job_id": job_id}
    guard.save(directory / "install-attempt.json", {"time": guard.now(), "source_head": head,
                                                   "actors": rows, "identities": identities, "driver": driver})
    guard.save(directory / "guard-contract.json", frozen)
    env = guard.environment(runtime)
    python = str(Path(env["VIRTUAL_ENV"]) / "bin/python")
    argv = [python, "-u", "-B", str(ST / "exit_guard.py"), "--guard-dir", str(directory)]
    with (directory / "guard.log").open("x") as stream:
        child = subprocess.Popen(argv, cwd=ST, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    identity = guard.proc(child.pid)
    guard.save(directory / "spawn-receipt.json", {"time": guard.now(), "identity": identity, "argv": argv})
    for _ in range(60):
        if (directory / "guard-ready.json").exists():
            break
        guard.require(child.poll() is None, "Guard exited before readiness")
        time.sleep(0.25)
    guard.require((directory / "guard-ready.json").exists(), "Guard did not publish readiness")
    ready = guard.load(directory / "guard-ready.json")
    guard.require(ready["passed"] and ready["namespace"] == contract["namespace"] and ready["job_id"] == job_id, "Guard readiness target mismatch")
    guard.require(all(ready["identity"][key] == identity[key] for key in ("pid", "uid", "start")) and guard.same(identity), "Guard process identity changed")
    guard.require(ready["guard_contract_sha256"] == guard.sha(directory / "guard-contract.json"), "Guard contract changed")
    receipt = {"time": guard.now(), "passed": True, "run": contract["run"], "namespace": contract["namespace"],
               "job_id": job_id, "guard_identity": identity, "driver": driver, "guard_script_sha256": script_hash,
               "guard_directory": str(directory), "source_head": head,
               "runtime_contract_sha256": frozen["runtime_contract_sha256"]}
    guard.save(directory / "install-receipt.json", receipt)
    return receipt


def main():
    guard.require(os.getuid() == 20001, "Install as chenyiteng UID20001")
    host = {"h100-gpu02": "sz2", "h100-gpu01": "sz3"}[socket.gethostname()]
    plan = guard.load(ST / "plans.json")[host]
    guard.require(len(plan["runs"]) == 2 and plan["hostname"] == socket.gethostname(), "Wrong host/run plan")
    guard.require(not (ST / "guard-install-all-attempt.json").exists(), "Guard installation already attempted on this host")
    guard.save(ST / "guard-install-all-attempt.json", {"time": guard.now(), "host": host,
                                                     "installer_sha256": guard.sha(__file__),
                                                     "guard_script_sha256": guard.sha(ST / "exit_guard.py")})
    receipts = [install(run_spec) for run_spec in plan["runs"]]
    result = {"time": guard.now(), "host": host, "passed": True, "runs": receipts}
    guard.save(ST / "guard-install-all-receipt.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
