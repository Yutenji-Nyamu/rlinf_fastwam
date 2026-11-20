"""Read actual, completed native episodes and show the active formal-run phase."""
import csv
import datetime
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.is_file() else []

def write(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")

cards = [5, 6]
gpus = {}
waves = {}
for card in cards:
    folder = root / f"native-g{card}"
    progress = read(folder / "progress.json")
    audits = read(folder / "reset-audit.json")
    counts = progress.get("counts", [])
    waves[card] = rows(folder / "completed-waves.jsonl")
    gpus[card] = dict(phase=progress.get("phase"), mode=progress.get("mode", audits.get("mode")),
                      slots=len(audits.get("layout_ids", [])),
                      action_range=([min(counts[:len(audits.get("layout_ids", []))] or counts), max(counts)] if counts else []),
                      completed_waves=len(waves[card]), group=audits.get("group_size"))

mem_path = root / "gpu-samples.csv"
mem = {g: [] for g in cards}
if mem_path.exists():
    for ts, gpu, used, total, utilization in csv.reader(mem_path.read_text().splitlines()):
        mem[int(gpu)].append(int(used))
for gpu in cards:
    if mem[gpu]:
        gpus[gpu].update(latest_GiB=round(mem[gpu][-1] / 1024, 1),
                         peak_GiB=round(max(mem[gpu]) / 1024, 1))

standard = rows(root / "standard-evaluations.jsonl")
full = rows(root / "evaluations.jsonl")
baseline = {}
if standard and full and standard[0]["round"] == full[0]["round"] == 0:
    first = standard[0]
    native_eval = [[w for w in waves[g] if w["mode"] == "eval"][0] for g in cards]
    results = [r for w in native_eval for r in w["rows"]]
    baseline = dict(task="pour_liquid_into_cup", protocol="standard, official seed-1 policy, before learning",
                    successes_25=first["successes"], independent_layouts_25=first["episodes"],
                    successes_32=sum(r["success"] for r in results), evaluated_32=len(results),
                    layout_outcomes_25=first["rows"], layout_outcomes_32=results, internal_metrics=full[0]["metrics"])
    assert len(results) == 32 and len({r["layout"] for r in results}) == 32
    write(root / "baseline-public.json", baseline)
    print("baseline", {k:v for k,v in baseline.items() if k in ("successes_25","successes_32")})

group_stats = []
for gpu in cards:
    for w in waves[gpu]:
        if w["mode"] != "train":
            continue
        r = w["rows"]
        for i in range(0, len(r), 8):
            grp = r[i:i+8]
            assert len(grp) == 8 and len({x["layout"] for x in grp}) == 1
            group_stats.append(sum(x["success"] for x in grp))

if baseline and group_stats:
    sampling = dict(task="pour_liquid_into_cup", simultaneous_envs=64, envs_per_gpu=32,
                    trajectories_per_formal_round=256, group_size=8,
                    completed_training_groups=len(group_stats),
                    completed_training_episodes=len(group_stats)*8,
                    observed_successes=sum(group_stats),
                    groups_with_both_success_and_failure=sum(0<x<8 for x in group_stats),
                    sampled_device_peak_GiB={str(g):gpus[g]["peak_GiB"] for g in cards},
                    memory_sample_period_s=15)
    write(root / "sampling-receipt.json", sampling)
    print("training", sampling)

print(datetime.datetime.now().astimezone().isoformat(timespec="seconds"),read(root/"status.json").get("phase"), gpus)
