"""Fixed-observation, per-action K4 entropy; reuse the writing-side mask."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def entropy(actions, epsilon=1e-6):
    """[B,K,H,D] -> [B,H], diagonal Gaussian estimate, normalized real D14."""
    a = np.asarray(actions, dtype=np.float64)[..., :14]
    if a.ndim != 4 or a.shape[1] < 2 or a.shape[-1] != 14:
        raise ValueError("Entropy needs >=2 samples of the same policy, D14")
    if not np.isfinite(a).all() or epsilon <= 0:
        raise ValueError("Invalid actions or epsilon")
    return 0.5 * np.log(2 * np.pi * np.e * (a.var(axis=1, ddof=1) + epsilon)).sum(-1)


def reached_positions(batch):
    """Writing revision_03: capped ceil of reached arm projection, not submitted H."""
    episodes = json.loads((Path(batch) / "episodes.json").read_text())
    reached = {}
    for e in episodes:
        frames = json.loads((Path(batch) / e["frame_map"]).read_text())["frames"]
        for f in frames:
            q = f.get("query", -1)
            vals = [f.get(k) for k in ("left_h_projection", "right_h_projection")]
            vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
            if q >= 0 and vals:
                key = (e["slot"], q)
                reached[key] = max(reached.get(key, -1.0), max(vals))
    return episodes, reached


def executed_count(reached, slot, query, before, after, active):
    offered = int(after - before) if active else 0
    return min(offered, max(0, int(np.ceil(reached.get((slot, query), -1))) + 1))


def comparison(base, final, n_bins=10):
    """Equal-count global top20, stable ties; groups fixed by base entropy."""
    base, final = np.asarray(base), np.asarray(final)
    if base.shape != final.shape or base.ndim != 1 or len(base) < 10:
        raise ValueError("Expected matching fixed-pool positions")
    if not np.isfinite(base).all() or not np.isfinite(final).all():
        raise ValueError("Incomplete fixed pool")
    k = max(1, int(np.ceil(len(base) * 0.2)))
    ib = np.argsort(-base, kind="stable")[:k]
    it = np.argsort(-final, kind="stable")[:k]
    bins = []
    if not 1 <= n_bins <= len(base):
        raise ValueError("Invalid entropy group count")
    for i, ids in enumerate(np.array_split(np.argsort(base, kind="stable"), n_bins)):
        bins.append(
            dict(
                group=i + 1,
                n=len(ids),
                base=float(base[ids].mean()),
                final=float(final[ids].mean()),
                delta=float((final[ids] - base[ids]).mean()),
            )
        )
    return dict(
        n=len(base),
        top_k=k,
        overlap=float(np.intersect1d(ib, it).size / k),
        random_reference=k / len(base),
        bins=bins,
        base_quantiles=np.quantile(base, [0.5, 0.8, 0.95]).tolist(),
        final_quantiles=np.quantile(final, [0.5, 0.8, 0.95]).tolist(),
    )


def analyze(batch, output, teacher=None):
    batch, output = Path(batch), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    episodes, reached = reached_positions(batch)
    rows = []
    for f in sorted(batch.glob("query_*.npz")):
        q = int(f.stem.split("_")[-1])
        with np.load(f, allow_pickle=False) as z:
            a = z["noise4_model_actions"]
            hs, h3 = entropy(a), entropy(a[:, 1:])
            ht = None
            if teacher:
                with np.load(Path(teacher) / f.name, allow_pickle=False) as t:
                    np.testing.assert_array_equal(
                        z["teacher_model_actions"], t["teacher4_model_actions"][:, 0]
                    )
                    ht = entropy(t["teacher4_model_actions"])
            for e in sorted(episodes, key=lambda e: e["slot"]):
                s = e["slot"]
                n = executed_count(
                    reached, s, q, z["before"][s], z["after"][s], z["active"][s]
                )
                for h in range(n):
                    rows.append(
                        dict(
                            slot=s,
                            seed=e["actual"],
                            query=q,
                            h=h,
                            action=int(z["before"][s]) + h,
                            success=e["success"],
                            student_entropy=float(hs[s, h]),
                            teacher_entropy=float(ht[s, h]) if ht is not None else "",
                            student_entropy_without_main=float(h3[s, h]),
                            distance=float(z["teacher_student_rmse"][s, h]),
                            relative_dispersion=float(z["noise4_u"][s, h]),
                        )
                    )
    if not rows:
        raise ValueError("No valid executed positions")
    with (output / "positions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = dict(
        episodes=len(episodes),
        successes=sum(e["success"] for e in episodes),
        positions=len(rows),
        input_batch=str(batch),
        teacher_supplement=str(teacher) if teacher else None,
        formula="diagonal Gaussian; sample variance ddof=1; epsilon=1e-6; normalized D14",
        mask="writing revision_03: ceil(max reached arm projection)+1, capped by submitted prefix",
        pool_policy="Clean CP140; not a final-policy pool of a single training run",
    )
    if teacher:
        result["paired_comparison"] = comparison(
            [r["student_entropy"] for r in rows], [r["teacher_entropy"] for r in rows]
        )
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("batch")
    p.add_argument("output")
    p.add_argument("--teacher")
    args = p.parse_args()
    print(json.dumps(analyze(args.batch, args.output, args.teacher)))
