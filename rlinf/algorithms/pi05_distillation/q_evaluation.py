"""Optional frozen-Q evaluation on the policy's existing prefix and actions."""

import json
from pathlib import Path

import numpy as np
import torch

from .fql import TwinQ, prefix_mask


def pooled_features(prefix, valid, state):
    pooled = torch.bmm(valid[:, None].to(prefix.dtype), prefix).squeeze(1).float()
    pooled = pooled / valid.sum(1, keepdim=True).clamp_min(1)
    return torch.cat((pooled, state[:, :14].float()), dim=-1)


class QEvaluation:
    def __init__(self, path, config, device, policy_metadata):
        export = torch.load(path, map_location="cpu", weights_only=True)
        assert export["base"] == config["model"]["model_path"]
        assert export["chunk"] == config["execute_chunk"]
        assert (
            export["step"] == policy_metadata["step"]
            and export["base"] == policy_metadata["base"]
        )
        self.discount = export["discount"]
        self.chunk = export["chunk"]
        with torch.random.fork_rng(devices=[]):
            self.q = TwinQ(
                export["feature_dim"], self.chunk, 14, export["width"], export["depth"]
            )
        self.q.load_state_dict(export["q"], strict=True)
        self.q = self.q.to(device).eval().requires_grad_(False)
        self.records = []

    @torch.inference_mode()
    def predict(self, features, actions, counts, limit):
        remaining = torch.as_tensor(limit - counts, device=features.device)
        values = self.q(
            features, actions, prefix_mask(remaining, self.chunk), remaining / limit
        )
        return values.float().cpu().numpy()

    def record(
        self,
        values,
        seeds,
        query,
        active,
        before,
        after,
        success_before,
        success_after,
        limit,
    ):
        for slot in np.flatnonzero(active):
            self.records.append(
                {
                    "slot": int(slot),
                    "seed": int(seeds[slot]),
                    "query": int(query),
                    "before": int(before[slot]),
                    "after": int(after[slot]),
                    "q1": float(values[slot, 0]),
                    "q2": float(values[slot, 1]),
                    "reward": int(success_after[slot] and not success_before[slot]),
                    "terminal": bool(success_after[slot] or after[slot] >= limit),
                }
            )

    def save(self, output):
        records, summary = summarize(self.records, self.discount)
        output = Path(output)
        (output / "q-records.json").write_text(
            json.dumps(records, indent=2, allow_nan=False)
        )
        (output / "q-calibration.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False)
        )


def summarize(records, discount):
    """Finite episode return, using the same per-macro-action gamma as training."""
    rows = [dict(r) for r in records]
    assert rows
    groups = {}
    for row in rows:
        groups.setdefault(row["slot"], []).append(row)
    for episode in groups.values():
        episode.sort(key=lambda r: r["query"])
        assert episode[-1]["terminal"], "Do not score an incomplete episode"
        assert not any(r["terminal"] for r in episode[:-1])
        assert sum(r["reward"] for r in episode) <= 1
        value = 0.0
        for row in reversed(episode):
            value = row["reward"] + (0 if row["terminal"] else discount * value)
            row["actual_discounted_return"] = value
            row["q_mean"] = (row["q1"] + row["q2"]) / 2
    prediction = np.array([r["q_mean"] for r in rows])
    truth = np.array([r["actual_discounted_return"] for r in rows])
    assert np.isfinite(prediction).all() and np.all((truth >= 0) & (truth <= 1))
    edges = [-np.inf, 0, 0.2, 0.4, 0.6, 0.8, 1, np.inf]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (prediction >= lo) & (prediction < hi)
        if selected.any():
            bins.append(
                {
                    "lower": None if not np.isfinite(lo) else lo,
                    "upper": None if not np.isfinite(hi) else hi,
                    "count": int(selected.sum()),
                    "prediction_mean": float(prediction[selected].mean()),
                    "actual_mean": float(truth[selected].mean()),
                }
            )
    return rows, {
        "episodes": len(groups),
        "records": len(rows),
        "discount": discount,
        "mae": float(np.abs(prediction - truth).mean()),
        "bias": float((prediction - truth).mean()),
        "q_mean": float(prediction.mean()),
        "actual_return_mean": float(truth.mean()),
        "outside_0_1_fraction": float(((prediction < 0) | (prediction > 1)).mean()),
        "twin_disagreement_mean": float(
            np.mean([abs(r["q1"] - r["q2"]) for r in rows])
        ),
        "bins": bins,
        "aggregation": "macro-action-weighted; episodes are independent units",
    }
