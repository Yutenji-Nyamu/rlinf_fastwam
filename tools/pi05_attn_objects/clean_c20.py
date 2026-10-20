"""Clean CP140 weak-policy calibration -> 10 high + 10 low C20 conditions."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

R = Path(__file__).resolve().parents[2]
S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
sys.path[:0] = [str(S / "ops"), str(R / "tools/pi05_policy_eval"), str(R / "tools/pi05_followups")]
from common import exact_signal, gpus, proc, read, save
import prepare as inherited
from calibration_pool import collect, deduplicate
from threshold_sweep import conditions

D = Path("/data/chenyiteng/deployment-20261020/clean-weak-c20-sz1-v1")
O = Path("/data/chenyiteng/results/pi05-policy-eval/clean-weak-c20-20261020-v1")
WEAK = Path("/data/chenyiteng/models/protected-eval/pillbottle/clean_c17_cp140/full_weights.pt")
STRONG = WEAK.parent.parent / "grpo_u_tau3_cp70/full_weights.pt"
PY = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"
FRACTIONS = (.05, .95, .25, .75, .45, .55, .15, .85, .35, .65)


def configs(root, transaction, specs):
    inherited.make_stages(root, transaction, specs)
    for p in (root / "configs").glob("*.json"):
        c = read(p)
        if not isinstance(c, dict) or "model" not in c:
            continue
        c.update(weak_weights=str(WEAK), comparison_baseline="clean_weak",
                 collect_weak_u=c["policy"] != "strong")
        save(p, c)


def prepare():
    assert os.getuid() == 1003 and WEAK.is_file() and STRONG.is_file()
    assert not (D / "prepared.json").exists()
    inherited.R, inherited.D = R, D
    inherited.dirs(O)
    configs(O, D / "calibration", [
        dict(condition="clean_weak", policy="calibration", chunk=20),
        dict(condition="tau3_strong", policy="strong", chunk=20, weights=STRONG),
    ])
    spec = dict(time=time.time(), source=str(R), root=str(O), weak=str(WEAK), strong=str(STRONG),
                task="move_pillbottle_pad", H=50, C=20, N=10, num_envs_per_card=32,
                episodes_per_condition=128, anchors=2, sweep_conditions=20, total_episodes=2816,
                calibration="128 Clean CP140-only episodes, all valid decisions; no outcome selection",
                target_fractions=sorted(FRACTIONS), actual_takeover_rate="measured after each condition",
                stop="inherited 600s progress timeout / 3600s batch timeout; on failure original RLT",
                original_fallbacks_preserved=True)
    save(D / "prepared.json", spec)
    save(O / "manifest.json", spec)
    save(D / "sequence-launch.json", dict(argv=[PY, "-u", "-B", str(Path(__file__)), "run"],
                                        cwd=str(R), env={"CUDA_VISIBLE_DEVICES": ""}))
    print(json.dumps(spec))


def run():
    me = proc(os.getpid())
    save(D / "sequence-identity.json", me)
    children = []

    def stop(*_):
        for child in children:
            if child.poll() is None:
                exact_signal(proc(child.pid), signal.SIGTERM)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)

    def phase(name):
        children.clear()
        T = D / name
        save(D / "sequence-status.json", dict(time=time.time(), phase=name))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", POLICY_EVAL_ROOT=str(O),
                   POLICY_EVAL_TRANSACTION=str(T), POLICY_EVAL_UID="1003", POLICY_EVAL_GPUS="4,5")
        for g in (4, 5):
            with (T / f"lane-g{g}.log").open("x") as log:
                child = subprocess.Popen(["/usr/bin/python3", "-u", "-B", str(R / "tools/pi05_policy_eval/lane.py"), str(g)],
                         cwd=R, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
            children.append(child)
        while any(c.poll() is None for c in children):
            assert all(c.poll() in (None, 0) for c in children), f"{name} lane failed"
            time.sleep(10)
        assert all(c.returncode == 0 for c in children)
        assert all(read(T / f"lane-g{g}-released.json")["completed"] for g in (4, 5))
        assert all(not gpus()[g]["processes"] for g in (4, 5))

    try:
        phase("calibration")
        result = subprocess.run([PY, "-u", "-B", str(Path(__file__)), "calibrate"],
                                cwd=R, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                                capture_output=True, text=True, timeout=300)
        (D / "calibrate.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0
        phase("sweep")
        save(D / "sequence-released.json", dict(time=time.time(), completed=True, identity=me))
        save(D / "sequence-status.json", dict(time=time.time(), phase="COMPLETE_RLT_HANDOFF"))
    except BaseException:
        save(D / "sequence-error.json", dict(time=time.time(), error=traceback.format_exc()))
        stop()


def calibrate():
    inherited.R, inherited.D = R, D
    pool = collect([O], chunk=20)
    # Only the new weak-only calibration has U. Never mix historical original-policy U.
    assert len(pool["source_batches"]) == 8
    assert all("clean_weak_b" in row["source"] and row["state"][1][1] == str(WEAK) for row in pool["rows"])
    pool = deduplicate([pool])
    save(O / "calibration.json", pool)
    rows = conditions(pool["rows"], fractions=FRACTIONS)
    specs = []
    for row in rows:
        name = row["condition"].replace("cp180_", "clean_weak_")
        specs.append(dict(condition=name, policy=row["policy"], chunk=20, weights=STRONG,
                          threshold={row["policy"]: row["threshold"], **row}))
    configs(O, D / "sweep", specs)
    save(O / "sweep-manifest.json", dict(conditions=rows, unique_decisions=pool["unique_decisions"],
                                       new_episodes=2560, weak=str(WEAK), strong=str(STRONG)))


if __name__ == "__main__":
    {"prepare": prepare, "run": run, "calibrate": calibrate}[sys.argv[1]]()
