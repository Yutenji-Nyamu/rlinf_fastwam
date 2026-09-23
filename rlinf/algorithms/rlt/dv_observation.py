"""Read-only summaries of newly collected RLT teacher DV, before replay reuse."""
import json
import math
from pathlib import Path

import torch


def concentration(values):
    x = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not x.numel():
        return {}
    total = x.sum()
    p = x / total if total > 0 else torch.full_like(x, 1 / x.numel())
    n = x.numel()
    return {
        "top20_mass": float(p.topk(max(1, math.ceil(n * .2))).values.sum()),
        "ess_fraction": float(1 / (n * p.square().sum())),
        "entropy_normalized": float(-(p * p.clamp_min(1e-300).log()).sum() / math.log(n)) if n > 1 else 1.,
        "zero_mass": float(total == 0),
    }


@torch.no_grad()
def summarize_collected(trajectories, *, selected_l=3, horizon=10):
    """Each input is one valid simulator transition/chunk; exclude post-done padding."""
    rows, episode, missing, invalid = [], [], 0, 0
    for traj in trajectories:
        obs = traj.curr_obs or {}
        v = obs.get("teacher_dvac_v")
        if v is None:
            missing += 1
        else:
            x = v.detach().cpu().double().reshape(-1, 3, v.shape[-1])[:, (2, 3, 4).index(selected_l), :horizon].reshape(-1)
            if not torch.isfinite(x).all() or (x < 0).any():
                invalid += 1
            else:
                episode.append(x.tolist())
        done = traj.dones is not None and bool(traj.dones.bool().any())
        success = obs.get("episode_success")
        if done and episode:
            rows.append({"episode_index": len(rows), "complete": True, "success": bool(success.any()) if success is not None else None, "dv": episode})
            episode = []
    if episode:
        rows.append({"episode_index": len(rows), "complete": False, "success": None, "dv": episode})
    metrics = {"missing_chunks": float(missing), "invalid_chunks": float(invalid), "episode_count": float(len(rows))}
    for group in ("all", "success", "failure"):
        selected = rows if group == "all" else [r for r in rows if r['success'] is (group == "success")]
        chunks = [x for r in selected for x in r['dv']]
        metrics[group + "/chunk_count"] = float(len(chunks))
        if not chunks:
            continue
        x = torch.tensor(chunks, dtype=torch.float64).flatten()
        for name, value in {"mean": x.mean(), "std": x.std(unbiased=False), "p50": x.quantile(.5), "p90": x.quantile(.9), "sum": x.sum()}.items():
            metrics[group + '/' + name] = float(value)
        for domain, vectors in [('chunk', chunks), ('episode', [sum(r['dv'], []) for r in selected])]:
            summaries = [concentration(v) for v in vectors]
            for name in summaries[0]:
                metrics[group + '/' + domain + '_' + name] = sum(v[name] for v in summaries) / len(summaries)
    return {'dv_observe/' + k: v for k, v in metrics.items()}, rows


def record_collected(trajectories, *, output_dir, round_number, rank, selected_l=3, horizon=10):
    metrics, episodes = summarize_collected(trajectories, selected_l=selected_l, horizon=horizon)
    path = Path(output_dir) / 'dv_observations' / f'rank_{rank:04d}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'schema_version': 1, 'round': round_number, 'selected_l': selected_l, 'horizon': horizon, 'source': 'frozen_teacher_new_collection', 'metrics': metrics, 'episodes': episodes}
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    return metrics
