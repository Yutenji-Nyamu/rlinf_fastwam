"""Apply three exact actor hooks; no broader upstream copy or algorithm changes."""

import argparse
from pathlib import Path


def patched_source(source):
    if "mask_invalid_rynn_groups" in source:
        raise ValueError("Actor already has a Rynn hook; do not patch twice")
    replacements = [
        ("        # filter data by rewards\n",
         '        self._rynn_reward_metrics = {}\n'
         '        if self.cfg.env.train.get("reward_source") == "rynn_success":\n'
         '            if self.cfg.env.train.get("rynn_invalid_reward_sentinel") != -1.0:\n'
         '                raise ValueError("Rynn invalid transport contract not configured")\n'
         '            from rlinf.envs.world_model.rynn_actor_mask import mask_invalid_rynn_groups\n'
         '            self._rynn_reward_metrics = mask_invalid_rynn_groups(\n'
         '                rollout_batch, self.cfg.algorithm.group_size)\n\n'
         '        # filter data by rewards\n'),
        ('        return rollout_batch\n\n    @Worker.timer("actor/compute_adv")',
         '        if self.cfg.env.train.get("reward_source") == "rynn_success":\n'
         '            from rlinf.envs.world_model.rynn_actor_mask import rynn_effective_group_metrics\n'
         '            self._rynn_reward_metrics.update(rynn_effective_group_metrics(\n'
         '                rollout_batch, self.cfg.algorithm.group_size))\n'
         '        return rollout_batch\n\n    @Worker.timer("actor/compute_adv")'),
        ('        rollout_metrics = compute_rollout_metrics(self.rollout_batch)\n        return rollout_metrics',
         '        rollout_metrics = compute_rollout_metrics(self.rollout_batch)\n'
         '        rollout_metrics.update(getattr(self, "_rynn_reward_metrics", {}))\n'
         '        return rollout_metrics'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("Actor source differs at a required exact hook")
        source = source.replace(old, new, 1)
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("actor_file", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    changed = patched_source(args.actor_file.read_text(encoding="utf-8"))
    compile(changed, str(args.actor_file), "exec")
    if args.apply:
        args.actor_file.write_text(changed, encoding="utf-8")
    print("Rynn actor hooks validated" + (" and applied" if args.apply else " (read only)"))
