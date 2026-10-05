"""Optional native-evaluation sidecar: simulator labels and exact per-env K8 clips.

Enable only in a private evaluation copy via RYNN_BINARY_CAPTURE_DIR. This does
not change observations, rewards, termination, seeds or policy inference.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def frame(value):
    value = array(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("Native main image must be uint8 HWC RGB-contract pixels")
    return np.asarray(Image.fromarray(value).resize((320, 256), Image.Resampling.BOX)).copy()


class NativeBinaryRecorder:
    def __init__(self, env, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.identity = f"{os.getpid()}-{time.time_ns()}"
        self.generation = [0] * env.num_envs
        self.pending = {}
        self.mode = os.environ.get("RYNN_BINARY_CAPTURE_MODE", "binary_terminal")
        if self.mode not in ("binary_terminal", "reward_native"):
            raise ValueError("Unknown capture mode")

    def reset(self, obs, env_idx, env_seeds):
        indices = range(self.env.num_envs) if env_idx is None else array(env_idx).reshape(-1).tolist()
        seeds = array(env_seeds).reshape(-1).tolist()
        images = obs["main_images"]
        for i in indices:
            if (i in self.pending and not self.pending[i]["written"]
                    and self.pending[i]["action_steps"][-1] > 0):
                raise RuntimeError("Native episode reset before a verified success or full horizon")
            self.generation[i] += 1
            instruction = str(obs["task_descriptions"][i])
            if not instruction.strip():
                raise ValueError("Native task instruction is empty")
            self.pending[i] = dict(id=f"native-{self.identity}-env{i}-ep{self.generation[i]}",
                seed=int(seeds[i] if len(seeds) == self.env.num_envs else seeds[list(indices).index(i)]),
                env_index=i, instruction=instruction, written=False,
                frames=[frame(images[i])], action_steps=[0],
                source_frame_shape=list(array(images[i]).shape),
                native_frames=[array(images[i]).copy()] if self.mode == "reward_native" else [],
                native_success=[None], first_success_position=None)

    def observe(self, obs, infos):
        if "success" not in infos:
            raise RuntimeError("Native simulator did not expose per-environment success")
        success = array(infos["success"]).reshape(-1)
        elapsed = array(self.env.elapsed_steps).reshape(-1)
        if len(success) != self.env.num_envs or any(x not in (0, 1, False, True) for x in success):
            raise ValueError("Native success vector is invalid")
        for i in range(self.env.num_envs):
            item = self.pending[i]
            if item["written"]:
                continue
            if int(elapsed[i]) <= item["action_steps"][-1]:
                raise RuntimeError("Native capture action clock did not advance")
            item["frames"].append(frame(obs["main_images"][i]))
            item["action_steps"].append(int(elapsed[i]))
            item["native_success"].append(bool(success[i]))
            if self.mode == "reward_native":
                item["native_frames"].append(array(obs["main_images"][i]).copy())
            if bool(success[i]) and item["first_success_position"] is None:
                item["first_success_position"] = len(item["frames"]) - 1
            at_horizon = int(elapsed[i]) >= int(self.env.cfg.max_episode_steps)
            done = at_horizon or (bool(success[i]) and self.mode == "binary_terminal")
            if not done:
                continue
            endpoint = item["first_success_position"] if item["first_success_position"] is not None else len(item["frames"]) - 1
            positions = np.linspace(0, endpoint, 8, dtype=int)
            clip = np.stack([item["frames"][int(j)] for j in positions])
            record = {k: v for k, v in item.items() if k not in ("frames", "written", "native_frames")}
            record.update(kind="native_eval", episode_uid=item["id"], frames_key=item["id"],
                reference_success=item["first_success_position"] is not None, task_name=self.env.task_name,
                capture_mode=self.mode,
                label_origin="native_simulator_infos_success_first_hit_or_full_horizon",
                source="RoboTwinEnv._record_metrics sidecar before ignore_terminations/auto_reset",
                frame_indices=positions.tolist(),
                selected_action_steps=[item["action_steps"][int(j)] for j in positions],
                max_episode_steps=int(self.env.cfg.max_episode_steps),
                terminal_action_steps=item["action_steps"][endpoint],
                collection_action_steps=int(elapsed[i]), native_success_at_end=bool(success[i]),
                array_sha256=hashlib.sha256(clip.tobytes()).hexdigest(),
                pixel_contract="native main_images; no channel permutation; PIL BOX to 320x256")
            output = self.root / item["id"]
            arrays = dict(frames=clip)
            if self.mode == "reward_native":
                arrays["native_frames"] = np.stack(item["native_frames"])
                record["native_frames_sha256"] = hashlib.sha256(arrays["native_frames"].tobytes()).hexdigest()
                record["native_frames_contract"] = "exact unresized native main_images; no channel permutation"
                record["initial_label_note"] = "reset observation has no simulator success query; None excludes it from binary accuracy"
            np.savez_compressed(output.with_suffix(".npz"), **arrays)
            tmp = output.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, output.with_suffix(".json"))
            item["written"] = True


def capture_reset(env, obs, env_idx, env_seeds):
    root = os.environ.get("RYNN_BINARY_CAPTURE_DIR")
    if root:
        if not hasattr(env, "_rynn_binary_recorder"):
            env._rynn_binary_recorder = NativeBinaryRecorder(env, root)
        env._rynn_binary_recorder.reset(obs, env_idx, env_seeds)


def capture_observe(env, obs, infos):
    if os.environ.get("RYNN_BINARY_CAPTURE_DIR"):
        env._rynn_binary_recorder.observe(obs, infos)
