"""Reuse the accepted RoboTwin U/noise configs and native SZ3 graphics binding."""
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

R = Path(__file__).resolve().parents[2]
S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
sys.path.insert(0, str(S / "ops"))
from common import read, save, gpus

D = Path("/data/chenyiteng/deployment-20261020/attn-objects-sz3-v1")
O = Path("/data/chenyiteng/results/pi05-attn-objects/20261020-v1")
OLD = Path("/data/chenyiteng/results/pi05-attn/20261010-video-v4")
assert os.getuid() == 20001 and not O.exists()
assert read(D / "old-stopped.json")["intentional_user_stop"]
assert not any(gpus()[g]["processes"] for g in (6, 7))
I = D / "inherited-u-configs"
I.mkdir()
with tarfile.open("/data/chenyiteng/exports/attn1020_uv_configs.tar.gz") as t:
    assert all(not n.startswith("/") and ".." not in Path(n).parts for n in t.getnames())
    t.extractall(I)
for sub in ("configs", "batches"):
    (O / sub).mkdir(parents=True)
jobs = {6: [], 7: []}
configs = [read(p) for p in sorted((I / "configs").glob("*.json")) if not p.name.endswith("_seeds.json")]
assert len(configs) == 100 and len({c["task"] for c in configs}) == 50
for c in configs:
    g = 6 + c["batch"]
    name = f"{c['task']}_b{c['batch']}"
    out = O / "batches" / name
    seeds = O / "configs" / f"{name}_seeds.json"
    shutil.copyfile(I / "configs" / Path(c["env"]["seeds_path"]).name, seeds)
    c.update(gpu=g, output=str(out), acceptance=False, noise_samples=4, attention_objects=True,
             source_commit=(R / "SOURCE_REVISION").read_text().strip(),
             signal="native U10/5 + DV + four-noise consistency + direct/via-visual/total object attention")
    c["env"]["seeds_path"] = str(seeds)
    c["env"]["task_config"]["save_path"] = str(out / "env")
    c["env"]["video_cfg"].update(save_video=False, video_base_dir=str(out / "unused"))
    c["env"]["task_config"]["eval_video_log"] = False
    path = O / "configs" / f"{name}.json"
    save(path, c)
    jobs[g].append(dict(config=str(path), task=c["task"], batch=c["batch"]))
before = read(D / "cutover-started.json")["plan_before"]
for g in (6, 7):
    first = "turn_switch" if g == 6 else "adjust_bottle"
    jobs[g].sort(key=lambda row: (row["task"] != first, row["task"]))
    env = read(OLD / f"environment-g{g}.json")
    old_source = env["REPO_PATH"]
    env = {k: v.replace(old_source, str(R)) if isinstance(v, str) else v for k, v in env.items()}
    env.update(REPO_PATH=str(R), CUDA_VISIBLE_DEVICES=str(g), DOJO_GPU_SCOPE=str(g),
               TMPDIR=str(D / f"tmp-g{g}"))
    Path(env["TMPDIR"]).mkdir(exist_ok=True)
    env.pop("RAY_ADDRESS", None)
    save(O / f"environment-g{g}.json", env)
    predecessor = dict(before["slots"][str(g)], external_release=str(D / "old-stopped.json"))
    save(D / f"lane-g{g}-plan.json", dict(gpu=g, jobs=jobs[g], predecessor=predecessor))
manifest = dict(time=time.time(), source=str(R), tasks=50, episodes=1600, batches=100,
                episodes_per_task=32, per_card_parallelism=16, physical_gpus=[6, 7],
                plan_before=before, original="SZ1 pi05-u-video/20261017-v1 configs/seeds",
                graphics="SZ3 accepted pi05-attn/20261010-video-v4 per-card environment",
                noise_samples=4, extra_inference_batch=48, steps_per_sample=10,
                attention=dict(layers=[10, 14, 18], heads="all native", denoise_calls=10, H=50,
                               main_plot="last evaluation", raw_QK=True, full_prefix_Q=True,
                               final_velocity_hidden=True, object_roles="surface-language heuristic; raw tokens retained"),
                video="accepted U full head-camera video at 25Hz plus endpoints; wrists in per-query observations",
                skip="native task errors / 600s stalled / 3600s per batch; exact child cleanup; no automatic retries",
                acceptance="first real B16 batch per card, parity/QK reconstruction/full video; then remaining 98",
                next="original per-card RLT")
save(O / "manifest.json", manifest)
save(D / "prepared.json", manifest)
# Launch gates keep their identity across exec; owner release path stays per-card.
for g in (6, 7):
    save(D / f"lane-g{g}-launch.json", dict(
        argv=["/usr/bin/python3", "-u", "-B", str(R / "tools/pi05_u_video/lane.py"), str(g)],
        cwd=str(R), env={"CUDA_VISIBLE_DEVICES": "", "U_VIDEO_ROOT": str(O), "U_VIDEO_TRANSACTION": str(D)}))
print(json.dumps(manifest))
