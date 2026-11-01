# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""GAD: detached K4 action signals, bounded weighting and mixed priority sampling.

The frozen model, teacher ODE, optimizer and baseline loss remain unchanged.
See docs/gad/09_IMPLEMENTATION_CONTRACT_20261101.md for provenance and choices.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist


def midranks(x):
    """Tie-aware empirical percentiles: constants are exactly 0.5, mean is 0.5."""
    shape = x.shape
    flat = x.detach().float().flatten()
    if not torch.isfinite(flat).all():
        raise FloatingPointError("Nonfinite GAD signal")
    _, inverse, counts = torch.unique(
        flat, sorted=True, return_inverse=True, return_counts=True
    )
    ends = counts.cumsum(0).float()
    centers = (ends - counts.float() / 2) / len(flat)
    return centers[inverse].reshape(shape)


def action_weights(student, teacher):
    """[B,K,H,D] normalized physical actions -> [B,H] weights and block priority.

    Population standard deviation (K=4), averaged over physical coordinates.
    Paired RMS distance is averaged over the four MATCHED generation noises.
    Ranks remove the incomparable scales of the three signals. Final rank
    gives a controlled 0.5..1.5 action multiplier with mean one, without tails.
    """
    if student.shape != teacher.shape or student.ndim != 4 or student.shape[1] != 4:
        raise ValueError("GAD requires four paired samples")
    s, t = student.detach().float(), teacher.detach().float()
    us, ut = s.std(1, correction=0).mean(-1), t.std(1, correction=0).mean(-1)
    disagreement = (s - t).square().mean(-1).sqrt().mean(1)
    score = (midranks(disagreement) + midranks(us) + 1 - midranks(ut)) / 3
    weight = 0.5 + midranks(score)
    return (
        weight,
        score.mean(-1),
        {
            "gad_student_std": us.mean(),
            "gad_teacher_std": ut.mean(),
            "gad_disagreement": disagreement.mean(),
            "gad_weight_min": weight.min(),
            "gad_weight_max": weight.max(),
            "gad_weight_mean": weight.mean(),
            "gad_weight_std": weight.std(correction=0),
            "gad_weight_ess_fraction": weight.mean().square() / weight.square().mean(),
        },
    )


@torch.no_grad()
def observe(
    student, teacher, actions, context, *, teacher_steps, seed, retain_candidates=False
):
    """Reuse one frozen prefix, four matched noises, no gradient or RNG side effect.

    Four B-sized passes avoid replicating the large KV prefix K times. Main
    baseline noise/target remains independent and is NEVER replaced by a mean.
    The separate OT variant may retain these samples to select a complete target.
    """
    rng = torch.Generator(device=actions.device).manual_seed(int(seed))
    s, t, noises = [], [], []
    for _ in range(4):
        z = torch.randn(
            actions.shape, device=actions.device, dtype=actions.dtype, generator=rng
        )
        if retain_candidates:
            noises.append(z)
        one = torch.ones(len(z), device=z.device)
        s.append((z - student(z, one, context))[..., :14].float())
        t.append(teacher.endpoint(z, context, steps=teacher_steps)[..., :14].float())
    s, t = torch.stack(s, 1), torch.stack(t, 1)
    result = action_weights(s, t)
    if retain_candidates:
        return (*result, {"noise": torch.stack(noises, 1), "student": s, "teacher": t})
    return result


def weighted_mean(elements, weights=None):
    """Reduce coordinates first; preserve signed DMD/MFD surrogate gradients."""
    if weights is None:
        return elements.mean()
    if elements.ndim != 3 or weights.shape != elements.shape[:2]:
        raise ValueError("GAD weights must match action positions [B,H]")
    return (elements.mean(-1) * weights.detach()).mean()


class PriorityBatches:
    """One-step CPU lookahead with exact saved draw probabilities and resume.

    Sampling: uniform task, then 50% uniform windows + 50% rank priority^0.6.
    Full importance correction prevents counting window selection twice.
    Priority is latest block score, duplicate visits averaged; unseen=0.5.
    The next draw happens before the current update. Save that small pending
    draw alongside priorities, so resume is independent of thread timing.
    """

    def __init__(self, data, *, rank, world, batch, seed, start, workers=8, state=None):
        self.data, self.rank, self.world = data, rank, world
        self.batch, self.seed = batch, seed
        self.sizes = np.asarray(getattr(data, "sizes", [len(data)]), dtype=np.int64)
        self.offsets = np.r_[0, self.sizes.cumsum()]
        self.priority = np.full(len(data), 0.5, dtype=np.float64)
        self.seen = np.zeros(len(data), dtype=np.int64)
        self.decoder = ThreadPoolExecutor(workers)
        self.producer = ThreadPoolExecutor(1)
        if state:
            if state["sizes"] != self.sizes.tolist():
                raise ValueError("GAD resume dataset changed")
            self.priority[:] = state["priority"]
            self.seen[:] = state["seen"]
            draw = state["pending"]
            if draw["step"] != start:
                raise ValueError("GAD pending draw does not match resume step")
        else:
            draw = self.draw(start)
        self.pending = draw
        self.future = self.submit(draw)

    def draw(self, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, 64001]))
        ids = np.empty(self.batch, dtype=np.int64)
        ratio, probability = np.empty(self.batch), np.empty(self.batch)
        tasks = rng.integers(len(self.sizes), size=self.batch)
        for task in np.unique(tasks):
            selected = np.flatnonzero(tasks == task)
            lo, hi = self.offsets[task : task + 2]
            values = self.priority[lo:hi]
            # CPU tie-aware ranks; table is only 4738 entries in this experiment.
            _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
            r = (np.cumsum(counts) - counts / 2)[inv] / len(values)
            mass = r**0.6
            p = 0.5 / len(values) + 0.5 * mass / mass.sum()
            local = rng.choice(len(values), size=len(selected), p=p)
            ids[selected] = lo + local
            probability[selected] = p[local] / len(self.sizes)
            ratio[selected] = (1 / len(values)) / p[local]
        return {
            "step": step,
            "ids": ids.tolist(),
            "importance": ratio.tolist(),
            "probability": probability.tolist(),
        }

    def submit(self, draw):
        return self.producer.submit(
            self.data.batch, draw["ids"][self.rank :: self.world], executor=self.decoder
        )

    def next(self, step):
        if self.pending["step"] != step:
            raise ValueError("GAD draw order changed")
        batch, draw = self.future.result(), self.pending
        self.pending = self.draw(step + 1)
        self.future = self.submit(self.pending)
        return batch, draw

    def update(self, ids, scores):
        row = (list(map(int, ids)), scores.detach().float().cpu().tolist())
        rows = [None] * self.world
        if dist.is_initialized():
            dist.all_gather_object(rows, row)
        else:
            rows = [row]
        all_ids = np.asarray([i for r in rows for i in r[0]], dtype=np.int64)
        all_scores = np.asarray([s for r in rows for s in r[1]], dtype=np.float64)
        unique, inv, counts = np.unique(
            all_ids, return_inverse=True, return_counts=True
        )
        if not np.isfinite(all_scores).all():
            raise FloatingPointError("Nonfinite GAD priority")
        self.priority[unique] = np.bincount(inv, weights=all_scores) / counts
        self.seen[unique] += counts

    def state_dict(self):
        return {
            "version": 1,
            "sizes": self.sizes.tolist(),
            "priority": self.priority.tolist(),
            "seen": self.seen.tolist(),
            "pending": self.pending,
        }

    def close(self):
        self.producer.shutdown(wait=True, cancel_futures=True)
        self.decoder.shutdown(wait=True, cancel_futures=True)
