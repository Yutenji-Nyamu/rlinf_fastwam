"""Opt-in full 50x32 DSRL noise on the pinned ff64209 Dojo source export.

Preserves the repeated-noise path. Actor/Q/replay see all 1600 coordinates;
the frozen VLA still receives [B,50,32]. No SAC loss or reward changes.
"""

import ast
import json
import sys
from pathlib import Path


def replace(text, old, new):
    if new in text:
        return text
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new, 1)


def apply(root):
    root = Path(root)
    p = root / "rlinf/models/embodiment/openpi/openpi_action_model.py"
    text = p.read_text()
    text = replace(text, "    dsrl_num_q_heads: int = 10", "    dsrl_full_noise: bool = False\n    dsrl_num_q_heads: int = 10")
    text = replace(text,
        "                action_horizon=self.config.action_horizon,\n            ).to(dtype=_dsrl_dtype)",
        "                action_horizon=1 if self.config.dsrl_full_noise else self.config.action_horizon,\n            ).to(dtype=_dsrl_dtype)\n"
        "            self.dsrl_action_noise_net.probability_float32 = self.config.dsrl_full_noise\n"
        "            if self.config.dsrl_full_noise:\n"
        "                assert self.config.dsrl_action_noise_dim == self.config.action_horizon * self.config.action_dim\n")
    text = replace(text,
        "                noise_actions = noise_actions.unsqueeze(1).repeat(\n                    1, self.config.action_horizon, 1\n                )",
        "                if self.config.dsrl_full_noise:\n"
        "                    noise_actions = noise_actions.reshape(-1, self.config.action_horizon, self.config.action_dim)\n"
        "                else:\n"
        "                    noise_actions = noise_actions.unsqueeze(1).repeat(\n"
        "                        1, self.config.action_horizon, 1\n                    )")
    text = replace(text,
        "        return action_noise, logprobs, dist_params\n",
        "        if self.config.dsrl_full_noise:\n"
        "            action_noise = action_noise.reshape(-1, self.config.action_horizon, self.config.action_dim)\n"
        "        return action_noise, logprobs, dist_params\n")
    text = replace(text,
        "        if actions.dim() == 3:\n            actions = actions[:, 0, :]  # [B, action_horizon, dim] -> [B, dim]",
        "        if actions.dim() == 3:\n"
        "            actions = actions.flatten(1) if self.config.dsrl_full_noise else actions[:, 0, :]\n"
        "        assert actions.shape[-1] == self.config.dsrl_action_noise_dim")
    ast.parse(text); p.write_text(text)

    p = root / "rlinf/models/embodiment/modules/gaussian_policy.py"
    text = p.read_text()
    old = "        std = torch.exp(log_std)\n\n        if deterministic:"
    new = ("        if getattr(self, 'probability_float32', False):\n"
           "            # Sum 1600 log densities in FP32, while FSDP parameters stay bf16.\n"
           "            mean, log_std = mean.float(), log_std.float()\n"
           "        std = torch.exp(log_std)\n\n        if deterministic:")
    text = replace(text, old, new)
    ast.parse(text); p.write_text(text)

    p = root / "rlinf/data/storage/replay/dsrl_transition.py"
    text = p.read_text()
    text = replace(text,
        "    u_spec: dict[str, Any] | None = None,\n) -> dict[str, Any]:",
        "    u_spec: dict[str, Any] | None = None,\n    full_noise: bool = False,\n) -> dict[str, Any]:")
    start = text.index("    selected_actions = trajectory.actions[keep]")
    end = text.index("\n    curr_states =", start)
    old = text[start:end]
    if "if full_noise:" not in old:
        new = ("    selected_actions = trajectory.actions[keep]\n"
               "    if full_noise:\n"
               "        if selected_actions[0].numel() != latent_dim:\n"
               "            raise ValueError('Full DSRL replay must retain all noise coordinates')\n"
               "        latent = selected_actions.reshape(num_valid, latent_dim).to(torch.bfloat16).cpu().contiguous()\n"
               "    else:\n" + "\n".join("    " + line for line in old.splitlines()[1:]))
        text = text[:start] + new + text[end:]
    ast.parse(text); p.write_text(text)

    p = root / "rlinf/workers/actor/fsdp_sac_policy_worker.py"
    text = p.read_text()
    text = replace(text,
        "                    latent_dim=openpi_cfg.dsrl_action_noise_dim,",
        "                    latent_dim=openpi_cfg.dsrl_action_noise_dim,\n"
        "                    full_noise=openpi_cfg.get('dsrl_full_noise', False),")
    ast.parse(text); p.write_text(text)
    (root / "dojo-full-noise-source.json").write_text(json.dumps({
        "base": "ff64209f8ee668b35837ae6469509b7f0257b89d + accepted Dojo adapter",
        "flag": "dsrl_full_noise", "default": False,
        "full_latent": 1600, "vla_noise_shape": [50, 32],
        "replay_schema": 3, "method": "full-dimensional latent SAC; no repeat or shuffle"
    }, indent=2))


if __name__ == "__main__":
    apply(sys.argv[1])
