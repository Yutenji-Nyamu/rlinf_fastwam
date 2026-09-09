# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Success-filtered replay at the policy-query/command-chunk boundary.

RoboTwin interpolates the entire submitted command with TOPP. A command mask is
not a claim about which physical interpolation steps ran before early success.
"""

from pathlib import Path
import uuid

import torch


def masked_fm_loss(
    loss: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    if mask.shape != loss.shape or (mask.sum(dim=(1, 2)) == 0).any():
        raise ValueError(
            "SFT mask must match loss and contain valid targets per query."
        )
    query_loss = (loss * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
    if sample_weights is not None:
        if not isinstance(sample_weights, torch.Tensor):
            raise ValueError("Sample weights must be a finite nonnegative tensor [B].")
        weights = sample_weights.detach().to(device=loss.device, dtype=loss.dtype)
        if (
            weights.shape != query_loss.shape
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("Sample weights must be finite nonnegative [B].")
        # Already normalized on the complete Adam batch before microbatch split.
        # An all-zero microbatch is valid; do not divide by its weight sum.
        query_loss = query_loss * weights
    return query_loss.mean()


RABC_TASK_CAPACITY = 4096


def _rabc_images(observation: dict, num_envs: int) -> torch.Tensor:
    if not isinstance(observation, dict) or "main_images" not in observation:
        raise ValueError("RABC requires raw environment main_images.")
    images = torch.as_tensor(observation["main_images"])
    if (
        images.dtype != torch.uint8
        or images.ndim != 4
        or images.shape[0] != num_envs
        or images.shape[-1] != 3
        or min(images.shape[1:3]) < 1
    ):
        raise ValueError("RABC main_images must be uint8 [env,H,W,3] RGB.")
    return images.detach().cpu()


def snapshot_rabc_observation(observation: dict, num_envs: int) -> dict:
    """Freeze the query's raw environment RGB before stepping mutable buffers.

    The task is the existing environment instruction, before policy tokenization.
    This function is called only when online_bc.rabc.enabled is true.
    """
    images = _rabc_images(observation, num_envs)
    tasks = observation.get("task_descriptions")
    if not isinstance(tasks, (list, tuple)) or len(tasks) != num_envs:
        raise ValueError("RABC requires one environment task_descriptions string per env.")
    if any(not isinstance(task, str) or not task for task in tasks):
        raise ValueError("RABC environment instructions must be nonempty strings.")
    if any(len(task.encode("utf-8")) > RABC_TASK_CAPACITY for task in tasks):
        raise ValueError("RABC environment instruction exceeds 4096 UTF-8 bytes.")
    return {"main_images": images.clone(), "task_descriptions": list(tasks)}


class SuccessEpisodeCollector:
    """Collect complete episodes; never retain post-terminal policy queries."""

    def __init__(self, num_envs: int, *, rabc_enabled: bool = False):
        self.num_envs = num_envs
        self.rabc_enabled = rabc_enabled
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
        *,
        pre_observation: dict | None = None,
        post_observation: dict | None = None,
        final_observation: dict | None = None,
    ) -> None:
        commands = torch.as_tensor(commands).detach().cpu()
        if commands.ndim != 3 or commands.shape[0] != self.num_envs:
            raise ValueError("Online BC requires [env, command_horizon, action_dim].")
        success = torch.as_tensor(success, dtype=torch.bool).reshape(self.num_envs)
        dones = (
            torch.as_tensor(dones, dtype=torch.bool).reshape(self.num_envs, -1).any(-1)
        )
        if self.rabc_enabled:
            before = snapshot_rabc_observation(pre_observation, self.num_envs)
            after_images = _rabc_images(post_observation, self.num_envs)
            terminal_images = (
                _rabc_images(final_observation, self.num_envs)
                if final_observation is not None
                else None
            )
            if dones.any() and terminal_images is None:
                raise ValueError("RABC done observations require the chunk's final_observation.")
            if before["main_images"].shape != after_images.shape or (
                terminal_images is not None and terminal_images.shape != after_images.shape
            ):
                raise ValueError("RABC pre/post/final RGB dimensions differ.")
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
            if self.rabc_enabled:
                task = before["task_descriptions"][i].encode("utf-8")
                task_tensor = torch.zeros(RABC_TASK_CAPACITY, dtype=torch.uint8)
                task_tensor[:len(task)] = torch.tensor(list(task), dtype=torch.uint8)
                pre_image = before["main_images"][i].clone()
                # Only the terminating env selects its terminal observation;
                # unfinished neighbours keep their real current post-query RGB.
                post_image = (terminal_images if bool(dones[i]) else after_images)[i].clone()
                if self.pending[i]:
                    first, previous = self.pending[i][0], self.pending[i][-1]
                    if not torch.equal(first["rabc_task_utf8"], task_tensor):
                        raise ValueError("RABC task instruction changed within one episode.")
                    if not torch.equal(previous["rabc_post_image"], pre_image):
                        raise ValueError("RABC query RGB boundaries are not contiguous.")
                    episode_key = first["rabc_episode_key"].clone()
                else:
                    # OS UUID does not consume policy/env torch or numpy RNG.
                    # Stored bytes survive retries/checkpoints and do not collide
                    # across env ranks, pipeline stages or fresh worker sessions.
                    episode_key = torch.tensor(list(uuid.uuid4().bytes), dtype=torch.uint8)
                record.update(
                    rabc_pre_image=pre_image,
                    rabc_post_image=post_image,
                    rabc_task_utf8=task_tensor,
                    rabc_task_length=torch.tensor(len(task), dtype=torch.int64),
                    rabc_episode_key=episode_key,
                )
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

    def __init__(
        self, seed: int, archive_path: str, max_success_chunks: int | None = None
    ):
        if max_success_chunks is not None and (
            type(max_success_chunks) is not int or max_success_chunks < 1
        ):
            raise ValueError("max_success_chunks must be null or a positive integer.")
        self.max_success_chunks = max_success_chunks
        self.filtered_success_episodes = 0
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
        # Admission only: reject the whole long success, never truncate its label.
        # Collector success metrics and DVAC moments have already been recorded.
        if self.max_success_chunks is not None:
            accepted = [ep for ep in episodes if len(ep) <= self.max_success_chunks]
            self.filtered_success_episodes += len(episodes) - len(accepted)
            episodes = accepted
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
        stats = {"success_episodes": self.episodes, "query_records": len(self)}
        if self.max_success_chunks is not None:
            stats["filtered_success_episodes"] = self.filtered_success_episodes
        return stats

    def save_checkpoint(self, save_path: str | Path) -> None:
        target = Path(save_path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                records=self.records,
                episodes=self.episodes,
                archive_id=self.archive_id,
                filtered_success_episodes=self.filtered_success_episodes,
                rng=self.rng.get_state(),
            ),
            target / "success_replay.pt",
        )

    def load_checkpoint(self, load_path: str | Path) -> None:
        state = torch.load(Path(load_path) / "success_replay.pt", weights_only=True)
        self.records = state["records"]
        self.episodes = state["episodes"]
        self.archive_id = state["archive_id"]
        # A configured limit applies to new admissions, not existing replay.
        self.filtered_success_episodes = state.get("filtered_success_episodes", 0)
        self.rng.set_state(state["rng"])
