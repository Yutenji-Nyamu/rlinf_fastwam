# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Complete query transitions and frozen remaining-time scoring for online IQL.

The old success collector/replay runs in parallel without modified admission or
sampling. All stored fields are tensors. Execution length -1 means unknown;
the full submitted 50x14 command remains the action, including terminal chunks.
"""

import base64
import hashlib
import io
import json
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import torch
from PIL import Image

SCHEMA_VERSION = 1
TASK_CAPACITY = 4096
REWARD_VERSION = "finite-budget-query-pbrs-v1"
OBS_FIELDS = {"main_images": "observation/image", "wrist_images": "observation/wrist_image", "states": "observation/state"}
NEXT_FIELDS = {"main_images": "iql_next_image", "wrist_images": "iql_next_wrist_image", "states": "iql_next_state"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bytes_tensor(value):
    return torch.tensor(list(value), dtype=torch.uint8)


def _text_tensor(value):
    raw = value.encode("utf-8")
    if not value.strip() or len(raw) > TASK_CAPACITY:
        raise ValueError("IQL requires a nonempty instruction of at most 4096 UTF-8 bytes.")
    result = torch.zeros(TASK_CAPACITY, dtype=torch.uint8)
    result[:len(raw)] = _bytes_tensor(raw)
    return result, torch.tensor(len(raw), dtype=torch.int64)


def snapshot_iql_observation(observation, num_envs):
    """Clone all three raw cameras and state before mutable environment stepping."""
    if not isinstance(observation, dict):
        raise ValueError("Missing raw IQL environment observation.")
    result = {}
    for key in OBS_FIELDS:
        if key not in observation or observation[key] is None:
            raise ValueError(f"Missing IQL observation field: {key}")
        value = torch.as_tensor(observation[key]).detach().cpu()
        if value.shape[0] != num_envs:
            raise ValueError(f"IQL {key} environment count mismatch.")
        if key == "states":
            if value.shape != (num_envs, 14) or not torch.isfinite(value).all():
                raise ValueError("IQL requires finite raw states [env,14].")
        else:
            ndim = 4 if key == "main_images" else 5
            if value.dtype != torch.uint8 or value.ndim != ndim or value.shape[-1] != 3:
                raise ValueError("IQL cameras require raw uint8 HWC RGB.")
            if min(value.shape[-3:-1]) < 1 or (key == "wrist_images" and value.shape[1] != 2):
                raise ValueError("IQL requires two nonempty wrist cameras.")
        result[key] = value.clone()
    tasks = observation.get("task_descriptions")
    if not isinstance(tasks, (list, tuple)) or len(tasks) != num_envs:
        raise ValueError("IQL requires one raw task description per environment.")
    for task in tasks:
        if not isinstance(task, str):
            raise ValueError("IQL task description must be a string.")
        _text_tensor(task)
    result["task_descriptions"] = list(tasks)
    return result


class TransitionEpisodeCollector:
    """Parallel collector for success AND failure, ending at the first boundary."""

    def __init__(self, num_envs, *, run_id, source_id="env", max_commands=200, auto_reset=False):
        if num_envs < 1 or max_commands < 1 or max_commands % 50:
            raise ValueError("IQL requires positive environments and a 50-aligned command budget.")
        self.num_envs, self.max_commands, self.auto_reset = num_envs, max_commands, bool(auto_reset)
        self.run_hash = canonical_hash({"run_id": str(run_id)})
        self.session_hash = canonical_hash({"run": self.run_hash, "source": str(source_id), "session": uuid.uuid4().hex})
        self.episode_ids = [0] * num_envs
        self.completed = []
        self.pending = [[] for _ in range(num_envs)]
        self.finished = [False] * num_envs

    def reset(self):
        if any(self.pending):
            raise ValueError("Cannot reset an unfinished IQL episode into a fabricated failure.")
        self.finished = [False] * self.num_envs

    def append(self, forward_inputs, commands, success, dones, versions=None, *,
               pre_observation, post_observation, final_observation=None,
               terminations=None, truncations=None):
        actions = torch.as_tensor(commands).detach().cpu()
        if actions.shape != (self.num_envs, 50, 14) or not torch.isfinite(actions).all():
            raise ValueError("IQL action must be the finite full submitted [env,50,14] proposal.")
        if versions is None or terminations is None or truncations is None:
            raise ValueError("IQL requires policy versions and raw termination/truncation flags.")
        before = snapshot_iql_observation(pre_observation, self.num_envs)
        after = snapshot_iql_observation(post_observation, self.num_envs)
        terminal = None if final_observation is None else snapshot_iql_observation(final_observation, self.num_envs)
        flags = lambda x: torch.as_tensor(x, dtype=torch.bool).reshape(self.num_envs, -1).any(-1)
        success, done, terminated, truncated = (flags(x) for x in (success, dones, terminations, truncations))
        if not torch.equal(done, terminated | truncated):
            raise ValueError("IQL done flags disagree with raw termination/truncation.")
        if self.auto_reset and done.any() and terminal is None:
            raise ValueError("Resetting IQL environments require true pre-reset final observations.")
        for i in range(self.num_envs):
            if self.finished[i]:
                continue
            pv = torch.as_tensor(versions[i]).detach().cpu()
            if not pv.numel() or not torch.isfinite(pv).all() or not (pv == pv.flatten()[0]).all():
                raise ValueError("IQL query requires one finite policy version.")
            version = float(pv.flatten()[0])
            if version < 0 or version != int(version):
                raise ValueError("IQL collection version must be a nonnegative integer.")
            q = len(self.pending[i])
            budget = (q + 1) * 50 >= self.max_commands
            end = bool(success[i] or done[i] or budget)
            selected = terminal if bool(done[i]) and terminal is not None else after
            row = {k: v[i].detach().cpu().clone() for k, v in forward_inputs.items()
                   if k.startswith("observation/") or k in ("tokenized_prompt", "tokenized_prompt_mask")}
            for env_key, actor_key in OBS_FIELDS.items():
                if actor_key not in row or not torch.equal(row[actor_key], before[env_key][i]):
                    raise ValueError(f"IQL raw pre-observation disagrees with actor field {actor_key}.")
            task, task_length = _text_tensor(before["task_descriptions"][i])
            if self.pending[i]:
                previous = self.pending[i][-1]
                for env_key, next_key in NEXT_FIELDS.items():
                    if not torch.equal(previous[next_key], before[env_key][i]):
                        raise ValueError("IQL pre/post query observations are not contiguous.")
                if not torch.equal(previous["iql_task_utf8"], task) or int(previous["iql_collection_round"]) != int(version) + 1:
                    raise ValueError("IQL task or collection policy changed inside an episode.")
                episode_key = previous["iql_episode_key"].clone()
            else:
                episode_key = _bytes_tensor(bytes.fromhex(canonical_hash({"session": self.session_hash, "env": i, "episode": self.episode_ids[i], "round": int(version) + 1})))
            row.update({NEXT_FIELDS[k]: selected[k][i].clone() for k in NEXT_FIELDS})
            reason = 1 if bool(success[i]) else 2 if budget or bool(truncated[i]) else 3 if bool(terminated[i]) else 0
            row.update(action=actions[i].flatten().clone(), action_valid_mask=torch.ones_like(actions[i], dtype=torch.bool),
                       query_idx=torch.tensor(q), episode_id=torch.tensor([i, self.episode_ids[i]]), policy_version=pv.clone(),
                       iql_schema_version=torch.tensor(SCHEMA_VERSION), iql_run_hash=_bytes_tensor(bytes.fromhex(self.run_hash)),
                       iql_episode_key=episode_key, iql_task_utf8=task, iql_task_length=task_length,
                       iql_collection_round=torch.tensor(int(version) + 1), iql_success=success[i].clone(),
                       iql_terminated=terminated[i].clone(), iql_truncated=truncated[i].clone(),
                       iql_episode_end=torch.tensor(end), iql_end_reason=torch.tensor(reason),
                       iql_budget_exhausted=torch.tensor(budget), iql_submitted_commands=torch.tensor((q + 1) * 50),
                       iql_actual_execution_length=torch.tensor(-1), iql_r_task=success[i].float(),
                       iql_bootstrap_mask=torch.tensor(0.0 if end else 1.0))
            self.pending[i].append(row)
            if end:
                self.completed.append(self.pending[i])
                self.pending[i] = []
                self.finished[i] = True
                self.episode_ids[i] += 1

    def drain(self):
        if any(self.pending):
            raise ValueError("IQL rollout ended with incomplete episodes; refusing fabricated terminal data.")
        result, self.completed = self.completed, []
        return result


def _episode_key(episode):
    if not episode:
        raise ValueError("Empty IQL episode.")
    keys = [bytes(row["iql_episode_key"].tolist()).hex() for row in episode]
    if len(set(keys)) != 1 or [int(r["query_idx"]) for r in episode] != list(range(len(episode))):
        raise ValueError("IQL episode identity/order mismatch.")
    if any(bool(r["iql_episode_end"]) for r in episode[:-1]) or not bool(episode[-1]["iql_episode_end"]):
        raise ValueError("IQL replay requires exactly one final episode boundary.")
    return keys[0]


def _episode_digest(episode):
    digest = hashlib.sha256()
    for row in episode:
        for key in sorted(row):
            value = row[key].detach().cpu().contiguous()
            digest.update(key.encode()); digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TransitionReplay:
    """Cumulative all-outcome replay with separate actor/critic sampling streams."""

    def __init__(self, seed, archive_path, *, actor_seed=None, contract=None):
        self.records, self.episode_digests = [], {}
        self.archive_path, self.archive_id = Path(archive_path), 0
        self.contract = dict(contract or {})
        self.critic_rng = torch.Generator().manual_seed(int(seed))
        self.actor_rng = torch.Generator().manual_seed(int(seed) + 1 if actor_seed is None else int(actor_seed))
        self.success_episodes = self.failure_episodes = 0
        self.scorer_identity = None

    def __len__(self):
        return len(self.records)

    def is_ready(self, min_buffer_size=1):
        return len(self) >= min_buffer_size

    def add_episodes(self, episodes):
        accepted, identities = [], {}
        scorer_identity = self.scorer_identity
        for episode in episodes:
            key = _episode_key(episode)
            for row in episode:
                if int(row["iql_schema_version"]) != SCHEMA_VERSION:
                    raise ValueError("IQL replay schema mismatch.")
                for field in ("iql_reward", "iql_r_shape", "iql_r_task", "iql_bootstrap_mask", "iql_remaining_before", "iql_remaining_after"):
                    if field not in row or not torch.isfinite(row[field]).all():
                        raise ValueError("IQL replay accepts only completely scored finite transitions.")
                identity = bytes(row["iql_scorer_identity"].tolist()).hex()
                if len(identity) != 64 or (scorer_identity is not None and identity != scorer_identity):
                    raise ValueError("IQL replay scorer/reward identity changed.")
                scorer_identity = identity
                if float(row["iql_bootstrap_mask"]) != (0.0 if bool(row["iql_episode_end"]) else 1.0):
                    raise ValueError("IQL bootstrap mask disagrees with the finite-budget terminal contract.")
                if float(row["iql_r_task"]) != float(bool(row["iql_success"])):
                    raise ValueError("IQL sparse reward disagrees with first success.")
            digest = _episode_digest(episode)
            known = identities.get(key, self.episode_digests.get(key))
            if known is not None:
                if known != digest:
                    raise ValueError("Duplicate IQL episode key has different content.")
                continue
            accepted.append(episode); identities[key] = digest
        if not accepted:
            return
        self.archive_path.mkdir(parents=True, exist_ok=True)
        path = self.archive_path / f"batch_{self.archive_id:06d}.pt"
        with path.open("xb") as handle:
            torch.save(accepted, handle)
        self.archive_id += 1
        self.scorer_identity = scorer_identity
        self.episode_digests.update(identities)
        self.records.extend(row for episode in accepted for row in episode)
        self.success_episodes += sum(bool(ep[-1]["iql_success"]) for ep in accepted)
        self.failure_episodes += sum(not bool(ep[-1]["iql_success"]) for ep in accepted)

    def _batch(self, indices):
        rows = [self.records[i] for i in indices]
        if not rows:
            raise ValueError("Cannot sample empty IQL replay.")
        return {"forward_inputs": {k: torch.stack([r[k] for r in rows]) for k in rows[0]}}

    def sample_actor(self, num_chunks):
        if not self.records or num_chunks < 1:
            raise ValueError("IQL actor sampling needs nonempty replay and a positive batch.")
        return self._batch(torch.randint(len(self), (num_chunks,), generator=self.actor_rng).tolist())

    def sample_critic(self, num_chunks):
        if not self.records or num_chunks < 1:
            raise ValueError("IQL critic sampling needs nonempty replay and a positive batch.")
        return self._batch(torch.randperm(len(self), generator=self.critic_rng)[:min(num_chunks, len(self))].tolist())

    def get_stats(self):
        return {"success_episodes": self.success_episodes, "failure_episodes": self.failure_episodes,
                "episodes": len(self.episode_digests), "query_records": len(self)}

    def save_checkpoint(self, save_path):
        path = Path(save_path); path.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": SCHEMA_VERSION, "contract": self.contract, "records": self.records,
                    "episode_digests": self.episode_digests, "archive_id": self.archive_id,
                    "scorer_identity": self.scorer_identity,
                    "success_episodes": self.success_episodes, "failure_episodes": self.failure_episodes,
                    "critic_rng": self.critic_rng.get_state(), "actor_rng": self.actor_rng.get_state()}, path / "transition_replay.pt")

    def load_checkpoint(self, load_path):
        state = torch.load(Path(load_path) / "transition_replay.pt", map_location="cpu", weights_only=True)
        if state["schema_version"] != SCHEMA_VERSION or state["contract"] != self.contract:
            raise ValueError("IQL replay schema/reward contract changed on resume.")
        self.records, self.episode_digests = state["records"], state["episode_digests"]
        self.archive_id = state["archive_id"]
        self.scorer_identity = state["scorer_identity"]
        self.success_episodes, self.failure_episodes = state["success_episodes"], state["failure_episodes"]
        self.critic_rng.set_state(state["critic_rng"]); self.actor_rng.set_state(state["actor_rng"])
        # Deliberately keep this instance's fresh attempt archive directory.


class IQLRynnValueClient:
    """Reuse the frozen scorer HTTP contract, retaining every pre/post camera.

Cache identity is specific to IQL reward/context semantics. Cached raw terminal
seconds are retained; absorbing potential zero is used only to construct reward.
"""

    def __init__(self, endpoint, cache_path, expected_revision, robot_description,
                 camera_description, timeout_seconds=900, *, gamma=.99, potential_scale=.1):
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("IQL frozen scorer must be a local HTTP service.")
        self.endpoint, self.cache_path = endpoint.rstrip("/"), Path(cache_path)
        self.timeout, self.gamma, self.potential_scale = float(timeout_seconds), float(gamma), float(potential_scale)
        self.robot_description, self.camera_description = str(robot_description), str(camera_description)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        health = self.request("/health")
        if health.get("ok") is not True or health["fingerprint"]["revision"] != expected_revision:
            raise ValueError("IQL scorer health/revision mismatch.")
        self.fingerprint = health["fingerprint"]
        self.identity = {"schema": SCHEMA_VERSION, "reward_version": REWARD_VERSION,
                         "context": "full-episode-query-boundaries-prefix-v1", "fingerprint": self.fingerprint,
                         "robot_description": self.robot_description, "camera_description": self.camera_description,
                         "gamma": self.gamma, "potential_scale": self.potential_scale}
        self.identity_hash = canonical_hash(self.identity)

    def request(self, route, body=None):
        request = urllib.request.Request(self.endpoint + route, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    @staticmethod
    def _png(image):
        image = torch.as_tensor(image).detach().cpu()
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("IQL scorer requires raw uint8 HWC RGB.")
        stream = io.BytesIO(); Image.fromarray(image.numpy(), mode="RGB").save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def score_episode(self, episode):
        from rlinf.algorithms.online_iql import shape_reward
        key = _episode_key(episode)
        instructions = [bytes(r["iql_task_utf8"][:int(r["iql_task_length"])].tolist()).decode("utf-8") for r in episode]
        if len(set(instructions)) != 1 or not instructions[0].strip():
            raise ValueError("IQL instruction changed or is missing.")
        for previous, current in zip(episode, episode[1:]):
            for env_key in OBS_FIELDS:
                if not torch.equal(previous[NEXT_FIELDS[env_key]], current[OBS_FIELDS[env_key]]):
                    raise ValueError("IQL episode camera/state boundary mismatch.")
        frames = [episode[0]["observation/image"]] + [r["iql_next_image"] for r in episode]
        body = {"episode_id": key, "instruction": instructions[0], "robot_description": self.robot_description,
                "camera_description": self.camera_description, "frames_png_b64": [self._png(frame) for frame in frames]}
        request_hash = canonical_hash(body)
        self.cache_path.mkdir(parents=True, exist_ok=True); path = self.cache_path / (key + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["identity"] != self.identity or cached["request_hash"] != request_hash:
                raise ValueError("IQL immutable score cache identity/content mismatch.")
            result = cached["result"]
        else:
            result = self.request("/score", body)
        if result.get("ok") is not True or result.get("episode_id") != key or result.get("fingerprint") != self.fingerprint:
            raise ValueError("IQL scorer failed or identity changed.")
        values = torch.tensor(result["remaining_seconds"], dtype=torch.float64)
        deltas = torch.tensor(result["delta_seconds"], dtype=torch.float64)
        if values.shape != (len(episode) + 1,) or deltas.shape != (len(episode),):
            raise ValueError("IQL scorer boundary count mismatch.")
        if not torch.isfinite(values).all() or not torch.isfinite(deltas).all() or (values < 0).any() or (values > 512).any():
            raise ValueError("IQL scorer remaining seconds outside finite official range.")
        if not torch.allclose(values[:-1] - values[1:], deltas, atol=1e-6, rtol=1e-6):
            raise ValueError("IQL scorer delta sign/alignment mismatch.")
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            with temporary.open("x") as stream:
                json.dump({"identity": self.identity, "request_hash": request_hash, "result": result}, stream, indent=2)
            temporary.replace(path)
        scored = []
        for i, source in enumerate(episode):
            row = dict(source)
            total = shape_reward(-values[i], -values[i + 1], row["iql_r_task"], row["iql_bootstrap_mask"], gamma=self.gamma, potential_scale=self.potential_scale)
            row.update(iql_remaining_before=values[i].clone(), iql_remaining_after=values[i + 1].clone(),
                       iql_reward=total.detach().cpu().clone(), iql_r_shape=(total - row["iql_r_task"]).detach().cpu(),
                       iql_scorer_identity=_bytes_tensor(bytes.fromhex(self.identity_hash)),
                       iql_score_request_hash=_bytes_tensor(bytes.fromhex(request_hash)))
            scored.append(row)
        return scored, values, key
