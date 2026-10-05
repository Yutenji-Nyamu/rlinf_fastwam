"""Remove explicitly marked unknown Rynn groups before reward statistics/GRPO."""


def mask_invalid_rynn_groups(batch, group_size):
    import torch

    rewards = batch["rewards"]
    if rewards.ndim != 3 or rewards.shape[1] % group_size:
        raise ValueError("Rynn requires [time, whole-G8 batch, action] rewards")
    if not torch.isfinite(rewards).all():
        raise ValueError("Nonfinite Rynn reward transport")
    allowed = (rewards == 0) | (rewards == 1) | (rewards == -1)
    if not allowed.all():
        raise ValueError("Rynn sparse transport only permits 0, 1, and reserved -1")
    invalid_rows = (rewards == -1).any(dim=0).any(dim=-1)
    invalid_groups = invalid_rows.reshape(-1, group_size).any(dim=1)
    invalid_members = invalid_groups.repeat_interleave(group_size)
    rewards = rewards.clone()
    rewards[:, invalid_members, :] = 0
    batch["rewards"] = rewards
    mask = batch.get("loss_mask")
    if mask is None:
        raise ValueError("Rynn needs the native done-derived loss mask")
    batch["loss_mask"] = mask & ~invalid_members[None, :, None]
    count = int(invalid_groups.sum().item())
    total = len(invalid_groups)
    return {"rynn_invalid_group_count": count,
            "rynn_group_count": total,
            "rynn_invalid_group_fraction": count / total if total else 0.0}


def rynn_effective_group_metrics(batch, group_size):
    mask = batch["loss_mask"]
    retained_members = mask.any(dim=0).any(dim=-1)
    groups = retained_members.reshape(-1, group_size)
    # The filter is group-level; differing done lengths may not split membership.
    if (groups.any(dim=1) != groups.all(dim=1)).any():
        raise ValueError("Rynn loss masking retained a partial GRPO group")
    retained = int(groups.all(dim=1).sum().item())
    return {"rynn_effective_group_count": retained,
            "rynn_effective_group_fraction": retained / len(groups) if len(groups) else 0.0}
