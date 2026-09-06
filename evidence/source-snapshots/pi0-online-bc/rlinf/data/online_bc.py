# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Success-filtered replay at the policy-query/command-chunk boundary.

RoboTwin interpolates the entire submitted command with TOPP. A command mask is
not a claim about which physical interpolation steps ran before early success.
"""

from pathlib import Path

import torch


def masked_fm_loss(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    if mask.shape != loss.shape or (mask.sum(dim=(1, 2)) == 0).any():
        raise ValueError(
            "SFT mask must match loss and contain valid targets per query."
        )
    return ((loss * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))).mean()


class SuccessEpisodeCollector:
    """Collect complete episodes; never retain post-terminal policy queries."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.completed = []
        self.episode_ids = [0] * num_envs
        self.reset()

    def reset(self) -> None:
        self.pending = [[] for _ in range(self.num_envs)]
        self.finished = [False] * self.num_envs

    def append(
        self,
        forward_inputs: dict[str, torch.Tensor],
        commands: torch.Tensor,
        success: torch.Tensor,
        dones: torch.Tensor,
        versions: torch.Tensor | None = None,
    ) -> None:
        commands = torch.as_tensor(commands).detach().cpu()
        if commands.ndim != 3 or commands.shape[0] != self.num_envs:
            raise ValueError("Online BC requires [env, command_horizon, action_dim].")
        success = torch.as_tensor(success, dtype=torch.bool).reshape(self.num_envs)
        dones = (
            torch.as_tensor(dones, dtype=torch.bool).reshape(self.num_envs, -1).any(-1)
        )
        for i in range(self.num_envs):
            if self.finished[i]:
                continue
            # Reuse pre-query observation/normalization inputs, not denoising
            # chains or model_action (the actual submitted command is the label).
            record = {
                k: v[i].detach().cpu().clone()
                for k, v in forward_inputs.items()
                if k.startswith("observation/")
                or k in ("tokenized_prompt", "tokenized_prompt_mask")
            }
            if not record:
                raise ValueError("Missing pre-query OpenPI observation inputs.")
            record["action"] = commands[i].flatten().clone()
            record["action_valid_mask"] = torch.ones_like(commands[i], dtype=torch.bool)
            record["query_idx"] = torch.tensor(len(self.pending[i]))
            record["episode_id"] = torch.tensor([i, self.episode_ids[i]])
            if versions is not None:
                record["policy_version"] = versions[i].detach().cpu().clone()
            self.pending[i].append(record)
            if bool(success[i]) or bool(dones[i]):
                if bool(success[i]):
                    self.completed.append(self.pending[i])
                self.pending[i] = []
                self.finished[i] = True
                self.episode_ids[i] += 1

    def drain(self) -> list[list[dict[str, torch.Tensor]]]:
        episodes, self.completed = self.completed, []
        return episodes


class SuccessReplay:
    """Cumulative query replay, with uniform replacement sampling and restart state."""

    def __init__(self, seed: int, archive_path: str):
        self.records = []
        self.episodes = 0
        self.archive_id = 0
        self.archive_path = Path(archive_path)
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.records)

    def is_ready(self, min_buffer_size: int = 1) -> bool:
        return len(self) >= min_buffer_size

    def add_episodes(self, episodes: list[list[dict[str, torch.Tensor]]]) -> None:
        if not episodes:
            return
        self.archive_path.mkdir(parents=True, exist_ok=True)
        archive = self.archive_path / f"batch_{self.archive_id:06d}.pt"
        # Exclusive creation prevents an accidental fresh run overwriting data.
        with archive.open("xb") as handle:
            torch.save(episodes, handle)
        self.archive_id += 1
        self.episodes += len(episodes)
        self.records.extend(record for episode in episodes for record in episode)

    def sample(self, num_chunks: int) -> dict:
        if not self.records:
            raise ValueError("Cannot sample an empty success replay.")
        indices = torch.randint(len(self), (num_chunks,), generator=self.rng).tolist()
        rows = [self.records[i] for i in indices]
        return {
            "forward_inputs": {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
        }

    def get_stats(self) -> dict[str, int]:
        return {"success_episodes": self.episodes, "query_records": len(self)}

    def save_checkpoint(self, save_path: str | Path) -> None:
        target = Path(save_path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                records=self.records,
                episodes=self.episodes,
                archive_id=self.archive_id,
                rng=self.rng.get_state(),
            ),
            target / "success_replay.pt",
        )

    def load_checkpoint(self, load_path: str | Path) -> None:
        state = torch.load(Path(load_path) / "success_replay.pt", weights_only=True)
        self.records = state["records"]
        self.episodes = state["episodes"]
        self.archive_id = state["archive_id"]
        self.rng.set_state(state["rng"])
