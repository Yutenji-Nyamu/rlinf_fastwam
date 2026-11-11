"""CPU-only existing K4 reconstruction and a score-independent K8 query pool."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .common import entropy, sha256, uniform_indices, write_json, write_npz


def queries(batch):
    # Observation archives match query_*.npz too; explicitly exclude them.
    return [p for p in sorted(batch.glob("query_*.npz")) if p.stem[6:].isdigit()]


def main(args):
    root, out = Path(args.input), Path(args.output)
    sys.path.insert(0, args.source)
    from tools.pi05_policy_eval.entropy_analysis import (
        executed_count,
        reached_positions,
    )

    selected = json.loads(Path(args.gallery).read_text())["selected"]
    episodes_index = []
    for episode in selected:
        batch, slot = root / "batches" / episode["batch_name"], episode["slot"]
        _, reached = reached_positions(batch)
        records, am, ae, nz, ids = [], [], [], [], []
        for path in queries(batch):
            q = int(path.stem[6:])
            with np.load(path, allow_pickle=False) as z:
                n = executed_count(
                    reached,
                    slot,
                    q,
                    z["before"][slot],
                    z["after"][slot],
                    z["active"][slot],
                )
                if n == 0:
                    continue
                a = z["noise4_model_actions"][slot : slot + 1]
                h = entropy(a)[0]
                h2 = entropy(a[:, :2])[0]
                h3 = entropy(a[:, :3])[0]
                raw_index = len(am)
                am.append(a[0])
                ae.append(z["noise4_env_actions"][slot])
                nz.append(z["noise4_initial_noise"][slot])
                ids.append(q)
                for t in range(n):
                    records.append(
                        dict(
                            query=q,
                            raw_index=raw_index,
                            h=t,
                            action=int(z["before"][slot]) + t,
                            entropy4=float(h[t]),
                            entropy3=float(h3[t]),
                            entropy2=float(h2[t]),
                        )
                    )
        assert records, episode["id"]
        target = out / "existing" / episode["id"]
        write_npz(
            target / "actions.npz",
            model_actions=np.stack(am),
            env_actions=np.stack(ae),
            initial_noise=np.stack(nz),
            queries=np.asarray(ids),
        )
        frames = json.loads((batch / episode["frame_map"]).read_text())
        meta = {
            k: episode[k]
            for k in [
                "id",
                "task",
                "batch_name",
                "slot",
                "actual",
                "success",
                "video",
                "frame_map",
            ]
        }
        meta.update(
            points=len(records),
            source=str(batch),
            formula="diagonal Gaussian; normalized D14; ddof1; eps1e-6",
            mask="capped ceil(max reached arm projection)+1",
            video_sha256=sha256(batch / episode["video"]),
        )
        write_json(target / "series.json", dict(metadata=meta, positions=records))
        write_json(target / "frame_map.json", frames)
        episodes_index.append(meta)
    write_json(out / "existing/index.json", episodes_index)

    pool, images, wrists, states, descriptions = [], [], [], [], []
    for task in ["turn_switch", "stack_bowls_two"]:
        batch = root / "batches" / (task + "_b0")
        eps = json.loads((batch / "episodes.json").read_text())
        qm = json.loads((batch / "queries.json").read_text())
        if isinstance(qm, dict):
            qm = qm.get("queries", qm)
        qby = {int(q.get("query", i)): q for i, q in enumerate(qm)}
        eligible = {}
        for path in queries(batch):
            q = int(path.stem[6:])
            with np.load(path, allow_pickle=False) as z:
                for slot in np.flatnonzero(z["active"]):
                    eligible.setdefault(int(slot), []).append(q)
        candidates = [e for e in eps if len(eligible.get(e["slot"], [])) >= 4]
        # Outcome-stratified when both outcomes exist; no uncertainty-based selection.
        yes = [e for e in candidates if e["success"]]
        no = [e for e in candidates if not e["success"]]
        picked = yes[:4] + no[:4]
        picked += [e for e in candidates if e not in picked][: 8 - len(picked)]
        assert len(picked) == 8, (task, len(picked))
        for e in picked:
            slot = e["slot"]
            qs = eligible[slot]
            for pos in uniform_indices(len(qs), 4):
                q = qs[pos]
                with np.load(batch / f"query_{q:03d}_obs.npz", allow_pickle=False) as z:
                    images.append(z["main_images"][slot])
                    wrists.append(z["wrist_images"][slot])
                    states.append(z["states"][slot])
                instruction = qby[q]["descriptions"][slot]
                assert isinstance(instruction, str) and instruction.strip()
                descriptions.append(instruction)
                pool.append(
                    dict(
                        index=len(pool),
                        task=task,
                        batch=str(batch),
                        query=q,
                        slot=slot,
                        seed=e["actual"],
                        success=e["success"],
                        instruction=instruction,
                    )
                )
    assert len(pool) == 64
    write_npz(
        out / "rt-pool/observations.npz",
        main_images=np.stack(images),
        wrist_images=np.stack(wrists),
        states=np.stack(states),
        task_descriptions=np.asarray(descriptions),
    )
    write_json(out / "rt-pool/queries.json", pool)
    write_json(
        out / "rt-pool/config.json",
        json.loads((root / "configs/turn_switch_b0.json").read_text()),
    )
    write_json(
        out / "cpu-preparation.json",
        dict(
            existing_episodes=len(episodes_index),
            new_observations=len(pool),
            pool_sha256=sha256(out / "rt-pool/observations.npz"),
            status="complete",
        ),
    )
    print(
        json.dumps(dict(existing=len(episodes_index), new_observations=len(pool))),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--gallery", required=True)
    main(p.parse_args())
