"""An RLinf environment for the isolated, local OpenDW + reward service.

All state lives on CPU. The service owns GPU models and synchronously offloads
them before acknowledging /offload. Only newly added env registration is needed.
This initial smoke contract supports whole C32 chunks, with auto-reset disabled.
"""

import io
import json
import uuid
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
import torch

from rlinf.envs.world_model.opendw_adapter import (
    compose_robotwin_views,
    map_frame_scores,
    next_command_state,
    prepare_raw_actions,
    resize_policy_views,
    split_robotwin_views,
)
from rlinf.envs.world_model.rynn_success import (
    INVALID_REWARD_SENTINEL, invalid_group_members, sample_history, validate_success_reply,
)


def resolve_service_route(cfg, seed_offset, total_num_processes):
    """Pinned EnvWorker passes rank*stage_num+stage_id, not a slot offset.

    2151a08 rlinf/workers/env/env_worker.py:420-429 also passes
    world_size*stage_num. For this pipeline1/two-rank run these are 0,1 and 2.
    """
    process_index, process_count = int(seed_offset), int(total_num_processes)
    if process_count < 1 or not 0 <= process_index < process_count:
        raise ValueError('Invalid EnvWorker process index/count')
    urls = cfg.get('service_urls', None)
    urls = list(urls) if urls else [cfg.service_url]
    multiple = bool(cfg.get('service_urls', None))
    if multiple and len(urls) != process_count:
        raise ValueError('service_urls must contain exactly one endpoint per EnvWorker/stage process')
    normalized, keys = [], []
    for value in urls:
        url = str(value).rstrip('/')
        parsed = urllib.parse.urlparse(url)
        if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment
                or (multiple and parsed.port is None)):
            raise ValueError('OpenDW service requires an explicit loopback HTTP host/port without path or credentials')
        normalized.append(url)
        host = '127.0.0.1' if parsed.hostname == 'localhost' else parsed.hostname
        keys.append((host, parsed.port or 80))
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate service endpoints would share one service between EnvWorkers')
    return normalized[process_index if multiple else 0], process_index, process_count


class OpenDWRobotwinEnv:
    def __init__(
        self, cfg, num_envs, seed_offset, total_num_processes,
        record_metrics=True, worker_info=None,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed = int(cfg.seed) + int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self.device = torch.device("cpu")
        self.group_size = int(cfg.group_size)
        self.chunk = int(cfg.get("chunk", 32))
        self.auto_reset = bool(cfg.get("auto_reset", False))
        self.ignore_terminations = bool(cfg.get("ignore_terminations", False))
        self.video_cfg = cfg.get("video_cfg", {})
        self.image_size = tuple(cfg.get("image_size", (256, 256)))
        self.use_fixed_reset_state_ids = bool(cfg.get("use_fixed_reset_state_ids", True))
        if self.chunk != 32 or int(cfg.get("frame_stride", 4)) != 4:
            raise ValueError("The Robotwin bundle smoke contract is C32 with frame stride 4")
        if self.num_envs % self.group_size:
            raise ValueError("Each environment process must contain whole GRPO groups")
        if self.auto_reset:
            raise ValueError("Initial OpenDW smoke uses auto_reset=False to preserve group starts")
        if int(cfg.max_episode_steps) % self.chunk:
            raise ValueError("Smoke max_episode_steps must contain whole C32 chunks")
        self.num_group = self.num_envs // self.group_size
        self.service_url, self.process_index, self.total_num_processes = resolve_service_route(
            cfg, seed_offset, total_num_processes)
        self.reward_source = str(cfg.get("reward_source", "worldarena"))
        if self.reward_source not in {"worldarena", "rynn_success"}:
            raise ValueError("Unknown reward source")
        self.use_rynn = self.reward_source == "rynn_success"
        self.rynn_url = None
        self.rynn_batch_size = int(cfg.get("rynn_batch_size", 1))
        if self.use_rynn:
            if self.ignore_terminations:
                raise ValueError("Rynn first-success contract requires ignore_terminations=False")
            if cfg.get("rynn_invalid_reward_sentinel") != INVALID_REWARD_SENTINEL:
                raise ValueError("Rynn requires the explicit actor invalid-group transport contract")
            urls = cfg.get("rynn_service_urls", [])
            if not urls:
                raise ValueError("Rynn requires per-rank service URLs")
            self.rynn_url = resolve_service_route(
                {"service_urls": urls}, seed_offset, total_num_processes)[0]
            batch_gate_path = cfg.get("rynn_batch_size_file")
            if batch_gate_path:
                gate = json.loads(Path(str(batch_gate_path)).read_text(encoding="utf-8"))
                if gate.get("passed") is not True:
                    raise ValueError("Rynn batch gate has not passed")
                self.rynn_batch_size = int(gate["selected_rm_batch"])
            if self.rynn_batch_size not in (1, 2, 4, 8, 16):
                raise ValueError("Unreviewed Rynn microbatch size")
        self._run_uid = str(cfg.get("rynn_run_id", uuid.uuid4().hex))
        self._episode_generation = -1
        self.timeout = float(cfg.get("request_timeout_s", 7200))
        with np.load(str(cfg.initial_state_path), allow_pickle=False) as source:
            self.dataset = {key: source[key].copy() for key in source.files}
        required = {"main_images", "wrist_images", "states", "instructions"}
        if not required.issubset(self.dataset):
            raise ValueError(f"Reset NPZ requires {sorted(required)}")
        count = len(self.dataset["states"])
        if count < 1 or self.dataset["states"].shape != (count, 14):
            raise ValueError("Reset states must be a nonempty [K,14] array")
        if not np.isfinite(self.dataset["states"]).all():
            raise ValueError("Non-finite reset states")
        if any(len(self.dataset[key]) != count for key in required):
            raise ValueError("Reset dataset arrays have different lengths")
        if self.dataset["wrist_images"].shape[1] != 2:
            raise ValueError("Reset wrist_images must contain left then right")
        if self.dataset["instructions"].dtype.kind not in {"U", "S"}:
            raise ValueError("Reset instructions must be strings without object pickles")
        self._generator = np.random.default_rng(self.seed)
        self._env_generators = [
            np.random.default_rng(np.random.SeedSequence([self.seed, env_index]))
            for env_index in range(self.num_envs)
        ]
        self.reset_state_ids = None
        self._is_start = True
        self._is_offloaded = True
        self._call_index = 0
        self.update_reset_state_ids()
        self._clear_runtime()

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = bool(value)

    def _clear_runtime(self):
        self.current_images = None
        self.policy_main_images = None
        self.policy_wrist_images = None
        self.states = None
        self.task_descriptions = []
        self.elapsed_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.prev_step_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self._terminated = np.zeros(self.num_envs, dtype=bool)
        self._truncated = np.zeros(self.num_envs, dtype=bool)
        self.episode_uids = np.asarray([""] * self.num_envs)
        self.head_histories = [[] for _ in range(self.num_envs)]
        self.rynn_invalid = np.zeros(self.num_envs, dtype=bool)
        self.rynn_last_success = np.full(self.num_envs, -2, dtype=np.int8)
        self.legacy_score_last = np.zeros(self.num_envs, dtype=np.float32)

    def update_reset_state_ids(self):
        if self.reset_state_ids is not None and self.use_fixed_reset_state_ids:
            return
        group_ids = self._generator.integers(0, len(self.dataset["states"]), size=self.num_group)
        self.reset_state_ids = np.repeat(group_ids, self.group_size)

    def reset(self, **kwargs):
        self._clear_runtime()
        self._episode_generation += 1
        self.states = self.dataset["states"][self.reset_state_ids].astype(np.float32, copy=True)
        self.current_images = np.stack([
            compose_robotwin_views(
                self.dataset["main_images"][index],
                self.dataset["wrist_images"][index, 0],
                self.dataset["wrist_images"][index, 1],
            )
            for index in self.reset_state_ids
        ])
        native_views = [
            resize_policy_views(
                self.dataset["main_images"][index],
                self.dataset["wrist_images"][index, 0],
                self.dataset["wrist_images"][index, 1],
                self.image_size,
            )
            for index in self.reset_state_ids
        ]
        self.policy_main_images = np.stack([views[0] for views in native_views])
        self.policy_wrist_images = np.stack([np.stack(views[1:]) for views in native_views])
        self.task_descriptions = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in self.dataset["instructions"][self.reset_state_ids]
        ]
        self.episode_uids = np.asarray([
            f"{self._run_uid}:p{self.process_index}:e{index}:r{self._episode_generation}"
            for index in range(self.num_envs)
        ])
        if self.use_rynn:
            self.head_histories = [[image[:256].copy()] for image in self.current_images]
        self._is_start = False
        return self._wrap_obs(), {}

    def _wrap_obs(self):
        return {
            "main_images": torch.from_numpy(self.policy_main_images.copy()),
            "wrist_images": torch.from_numpy(self.policy_wrist_images.copy()),
            "states": torch.from_numpy(self.states.copy()),
            "task_descriptions": list(self.task_descriptions),
        }

    def _post(self, endpoint, payload, content_type, service_url=None):
        request = urllib.request.Request(
            (service_url or self.service_url) + endpoint, data=payload,
            headers={"Content-Type": content_type}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def _control(self, endpoint, expected_offloaded):
        response = json.loads(self._post(endpoint, b"{}", "application/json"))
        if response.get("ok") is not True or response.get("is_offloaded") is not expected_offloaded:
            raise RuntimeError(f"OpenDW {endpoint} did not acknowledge completed state: {response}")
        self._is_offloaded = expected_offloaded

    def onload(self):
        # Always contact the service: another logical env may have offloaded it.
        self._control("/onload", False)

    def offload(self):
        # A synchronous reply is the GPU-unload barrier, not an async intention.
        try:
            self._control("/offload", True)
        finally:
            if self.use_rynn:
                response = json.loads(self._post("/offload", b"", "application/json", self.rynn_url))
                if response.get("ok") is not True or response.get("is_offloaded") is not True:
                    raise RuntimeError("Rynn did not acknowledge completed GPU offload")

    def _rynn_success(self, indices, future_heads):
        expected = (len(indices), 8, 256, 320, 3)
        if future_heads.shape != expected or future_heads.dtype != np.uint8:
            raise ValueError(f"Expected future head frames uint8{expected}")
        sampled, frame_indices = [], []
        for offset, index in enumerate(indices):
            history = self.head_histories[index]
            history.extend(frame.copy() for frame in future_heads[offset])
            if len(history) != 1 + (int(self.elapsed_steps[index]) + self.chunk) // 4:
                raise ValueError("Rynn history/action-time mismatch")
            frames, timestamps = sample_history(history)
            sampled.append(frames)
            frame_indices.append(timestamps)
        end_indices = self.elapsed_steps[indices] + self.chunk
        payload = io.BytesIO()
        np.savez(payload, frames=np.stack(sampled),
                 instructions=np.asarray(self.task_descriptions)[indices],
                 episode_uids=self.episode_uids[indices], end_action_indices=end_indices,
                 frame_action_indices=np.stack(frame_indices),
                 rm_batch_size=np.asarray(self.rynn_batch_size, dtype=np.int64),
                 request_id=np.asarray(f"{self._run_uid}-p{self.process_index}-call{self._call_index}"))
        response = json.loads(self._post("/infer", payload.getvalue(), "application/octet-stream", self.rynn_url))
        return validate_success_reply(response, self.episode_uids[indices], end_indices)

    def step(self, actions=None, **kwargs):
        raise NotImplementedError("Use chunk_step with a complete C32 action chunk")

    def chunk_step(self, policy_output_action):
        if self.current_images is None:
            raise RuntimeError("reset() must precede chunk_step()")
        actions = prepare_raw_actions(policy_output_action)
        if len(actions) != self.num_envs:
            raise ValueError("Action batch differs from num_envs")
        alive = ~(self._terminated | self._truncated)
        if self.ignore_terminations:
            alive = ~self._truncated
        rewards = np.zeros((self.num_envs, self.chunk), dtype=np.float32)
        first_success = np.full(self.num_envs, -1, dtype=np.int64)
        if alive.any():
            if np.any(self.elapsed_steps[alive] + self.chunk > int(self.cfg.max_episode_steps)):
                raise ValueError("Partial C32 chunk is outside this smoke contract")
            self.onload()
            indices = np.flatnonzero(alive)
            seeds = np.array([
                self._env_generators[index].integers(0, 2**31 - 1) for index in indices
            ], dtype=np.int64)
            payload = io.BytesIO()
            np.savez(
                payload, images=self.current_images[alive], actions=actions[alive],
                states=self.states[alive], seeds=seeds,
                instructions=np.asarray(self.task_descriptions)[alive],
                env_indices=indices, reset_ids=self.reset_state_ids[alive],
                global_env_indices=self.process_index * self.num_envs + indices,
                env_process_index=np.asarray(self.process_index, dtype=np.int64),
                env_process_count=np.asarray(self.total_num_processes, dtype=np.int64),
                request_id=np.asarray(f"seed{self.seed}-call{self._call_index}"),
                return_future_head_frames=np.asarray(self.use_rynn),
            )
            reply = self._post("/infer", payload.getvalue(), "application/octet-stream")
            with np.load(io.BytesIO(reply), allow_pickle=False) as result:
                next_images = result["next_images"].copy()
                scores = result["scores"].copy()
                future_heads = result["future_head_frames"].copy() if self.use_rynn else None
            if next_images.shape != (len(indices), 384, 320, 3) or next_images.dtype != np.uint8:
                raise ValueError("Service next_images must be uint8[B,384,320,3]")
            if scores.shape != (len(indices), 8) or not np.isfinite(scores).all():
                raise ValueError("Invalid legacy diagnostic scores")
            self.legacy_score_last[indices] = scores[:, -1]
            if self.use_rynn:
                decisions = self._rynn_success(indices, future_heads)
                self.rynn_last_success[indices] = decisions
                unknown = indices[decisions == -1]
                invalid = invalid_group_members(unknown, self.num_envs, self.group_size)
                self.rynn_invalid |= invalid
                # -1 is transport metadata, removed by the actor before reward filtering/GRPO.
                rewards[unknown, -1] = INVALID_REWARD_SENTINEL
                successes = indices[(decisions == 1) & ~invalid[indices]]
                rewards[successes, -1] = 1.0
                first_success[successes] = self.chunk
                self._terminated[successes] = True
                self._truncated |= invalid
                self._truncated[indices] |= self.elapsed_steps[indices] + self.chunk >= int(self.cfg.max_episode_steps)
            else:
                feedback = map_frame_scores(
                    scores, self.prev_step_reward[alive], self.elapsed_steps[alive],
                    max_episode_steps=self.cfg.max_episode_steps,
                    reward_coef=self.cfg.get("reward_coef", 1.0),
                    relative_reward=self.cfg.get("use_rel_reward", True),
                    success_threshold=self.cfg.get("success_reward_threshold", 0.9),
                )
            self.current_images[alive] = next_images
            generated_views = [split_robotwin_views(image, self.image_size) for image in next_images]
            self.policy_main_images[alive] = np.stack([views[0] for views in generated_views])
            self.policy_wrist_images[alive] = np.stack([np.stack(views[1:]) for views in generated_views])
            self.states[alive] = next_command_state(actions[alive])
            self.elapsed_steps[alive] += self.chunk
            if not self.use_rynn:
                self.prev_step_reward[alive] = feedback.next_score
                rewards[alive] = feedback.rewards
                first_success[alive] = feedback.first_success_action
                self._terminated[alive] |= feedback.terminated[:, -1]
                self._truncated[alive] |= feedback.truncated[:, -1]
            self._call_index += 1
        self.success_once |= self._terminated
        self.returns += np.maximum(rewards, 0).sum(axis=1) if self.use_rynn else rewards.sum(axis=1)
        terminated = np.zeros((self.num_envs, self.chunk), dtype=bool)
        truncated = np.zeros_like(terminated)
        if not self.ignore_terminations:
            terminated[:, -1] = self._terminated
        truncated[:, -1] = self._truncated
        obs = self._wrap_obs()
        info = {
            "episode": {
                "success_once": torch.from_numpy(self.success_once.copy()),
                "return": torch.from_numpy(self.returns.copy()),
                "episode_len": torch.from_numpy(self.elapsed_steps.copy()),
                "reward": torch.from_numpy(self.returns / np.maximum(self.elapsed_steps, 1)),
                "rynn_invalid": torch.from_numpy(self.rynn_invalid.copy()),
                "legacy_score_last": torch.from_numpy(self.legacy_score_last.copy()),
            },
            "first_success_action": torch.from_numpy(first_success),
            "reset_state_ids": torch.from_numpy(self.reset_state_ids.copy()),
            "rynn_last_success": torch.from_numpy(self.rynn_last_success.copy()),
        }
        return [obs], torch.from_numpy(rewards), torch.from_numpy(terminated), torch.from_numpy(truncated), [info]

    def get_state(self):
        if self.current_images is None:
            raise RuntimeError("No initialized environment state")
        buffer = io.BytesIO()
        np.savez(
            buffer, current_images=self.current_images, states=self.states,
            policy_main_images=self.policy_main_images, policy_wrist_images=self.policy_wrist_images,
            instructions=np.asarray(self.task_descriptions), elapsed_steps=self.elapsed_steps,
            prev_step_reward=self.prev_step_reward, success_once=self.success_once,
            returns=self.returns, terminated=self._terminated, truncated=self._truncated,
            reset_state_ids=self.reset_state_ids,
            rng=np.asarray(json.dumps(self._generator.bit_generator.state)),
            env_rng=np.asarray([json.dumps(rng.bit_generator.state) for rng in self._env_generators]),
            call_index=np.asarray(self._call_index), is_start=np.asarray(self._is_start),
            rynn_run_uid=np.asarray(self._run_uid), episode_generation=np.asarray(self._episode_generation),
            episode_uids=self.episode_uids, rynn_invalid=self.rynn_invalid,
            rynn_last_success=self.rynn_last_success, legacy_score_last=self.legacy_score_last,
            head_history_lengths=np.asarray([len(history) for history in self.head_histories], dtype=np.int64),
            head_history_frames=(np.concatenate([np.stack(history) for history in self.head_histories], axis=0)
                                 if self.use_rynn else np.empty((0, 256, 320, 3), dtype=np.uint8)),
        )
        return buffer.getvalue()

    def set_state(self, state):
        with np.load(io.BytesIO(state), allow_pickle=False) as source:
            for name in ("current_images", "policy_main_images", "policy_wrist_images", "states", "elapsed_steps", "prev_step_reward", "success_once", "returns", "reset_state_ids"):
                setattr(self, name, source[name].copy())
            self.task_descriptions = source["instructions"].tolist()
            self._terminated = source["terminated"].copy()
            self._truncated = source["truncated"].copy()
            self._generator.bit_generator.state = json.loads(str(source["rng"]))
            for generator, rng_state in zip(self._env_generators, source["env_rng"]):
                generator.bit_generator.state = json.loads(str(rng_state))
            self._call_index = int(source["call_index"])
            self._is_start = bool(source["is_start"])
            if "episode_uids" in source:
                self._run_uid = str(source["rynn_run_uid"])
                self._episode_generation = int(source["episode_generation"])
                for name in ("episode_uids", "rynn_invalid", "rynn_last_success", "legacy_score_last"):
                    setattr(self, name, source[name].copy())
                cursor = 0
                self.head_histories = []
                for length in source["head_history_lengths"]:
                    length = int(length)
                    self.head_histories.append([frame.copy() for frame in source["head_history_frames"][cursor:cursor + length]])
                    cursor += length
                if cursor != len(source["head_history_frames"]):
                    raise ValueError("Saved Rynn history lengths are inconsistent")
            elif self.use_rynn:
                raise ValueError("Cannot restore Rynn env without its episode history")

    def close(self):
        self.offload()
