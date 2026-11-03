"""Reuse accepted native Dojo main/config/render setup, replace only its policy loop.

One simulator owns all N environments on its physical GPU. RLinf sends a whole
action chunk, and native take_action_batch advances active environments together.
AF_UNIX lives in a mode-0700 directory with a private authentication key.
"""

import json
import os
import sys
import time
import traceback
import faulthandler
import signal
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "rlinf/envs/robodojo"))
from protocol import layout_pools, terminal_arrays, replace_invalid_layouts


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def image_hwc(obs, names):
    vision = obs.get("vision", {})
    for name in names:
        if name in vision:
            image = vision[name]
            if isinstance(image, dict):
                image = next(image[k] for k in ["color", "rgb"] if k in image)
            image = np.asarray(image)
            if image.shape[0] == 3 and image.shape[-1] != 3:
                image = image.transpose(1, 2, 0)
            if np.issubdtype(image.dtype, np.floating):
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            return image[..., :3].astype(np.uint8, copy=False)
    raise KeyError(f"Native camera missing: {names}")


def invalid_reset_slots(env, n):
    """Native layout stability marks success/end_flag without unstable_envs."""
    return {
        i for i in range(n)
        if i in env.unstable_envs or not env.success[i] or env.end_flag[i]
        or not env.scene_manager.layout_manager.layout_valid[i]
    }


class NativeLoop:
    def __init__(self, env, cfg):
        from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

        self.env, self.cfg = env, cfg
        self.pack, self.unpack = pack_robot_state, unpack_robot_state
        self.pools = layout_pools(cfg["layout_root"], cfg["task"])
        self.n = 0
        self.mode = None
        self.last_obs = None
        self.output = Path(cfg["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        # The accepted lean collection already disables streaming this way.
        env._stream_vision = lambda *args, **kwargs: None
        env.save_video = lambda *args, **kwargs: None
        self.totals = {
            "reset_seconds": 0.0,
            "action_seconds": 0.0,
            "observation_seconds": 0.0,
        }
        self.invalid_layouts = {
            mode: set(cfg.get("invalid_layouts", {}).get(mode, []))
            for mode in self.pools
        }

    def progress(self, phase, **extra):
        save(
            self.output / "progress.json",
            dict(
                time=time.time(),
                phase=phase,
                counts=self.env.take_action_cnt,
                **self.totals,
                **extra,
            ),
        )

    def observations(self):
        t = time.monotonic()
        rows = self.env.get_obs_batch(env_idx_list=list(range(self.n)), last_frame=True)
        result = {
            "main_images": np.stack(
                [
                    image_hwc(x, ["cam_high", "cam_head", "head_camera", "top_camera"])
                    for x in rows
                ]
            ),
            "wrist_images": np.stack(
                [
                    np.stack(
                        [
                            image_hwc(
                                x,
                                [
                                    "cam_left_wrist",
                                    "left_camera",
                                    "left_wrist",
                                    "wrist_left",
                                ],
                            ),
                            image_hwc(
                                x,
                                [
                                    "cam_right_wrist",
                                    "right_camera",
                                    "right_wrist",
                                    "wrist_right",
                                ],
                            ),
                        ]
                    )
                    for x in rows
                ]
            ),
            "states": np.stack(
                [
                    self.pack(
                        x, "joint", self.env.robot_action_dim_info, source_type="obs"
                    )
                    for x in rows
                ]
            ).astype(np.float32),
            "task_descriptions": [x["instruction"] for x in rows],
            "extra_view_images": None,
        }
        self.totals["observation_seconds"] += time.monotonic() - t
        return result

    def reset(self, request):
        requested_ids = list(request["layout_ids"])
        ids = list(requested_ids)
        self.n = len(ids)
        self.mode = request["mode"]
        group = int(request["group_size"])
        assert self.n <= self.env.num_envs
        assert all(len(set(ids[i : i + group])) == 1 for i in range(0, self.n, group))
        pool = self.pools[self.mode]
        ids, replacements = replace_invalid_layouts(
            ids,
            self.invalid_layouts[self.mode],
            len(pool),
            group,
            int(self.cfg["gpu"]) % 2,
        )
        # Exact accepted SeedManager mechanism; native layout loading is unchanged.
        self.env.seed_manager.seed_info = {i: dict(row) for i, row in enumerate(pool)}
        self.env.seed_manager.seed_list = list(range(len(pool)))
        self.env.seed_manager.st_idx = 0
        self.env.seed_manager.ed_idx = len(pool)
        self.env.seed_manager.idx = 0
        self.env.seed_manager._current_batch_seeds = None
        self.progress("reset_started", mode=self.mode, layout_ids=ids)
        start = time.monotonic()
        from utils.cluttered_generator import UnStableError

        for attempt in range(3):
            try:
                self.env.reset(ids)
            except UnStableError:
                if not invalid_reset_slots(self.env, self.n):
                    raise
            bad = invalid_reset_slots(self.env, self.n)
            if not bad:
                break
            failed = {ids[i] for i in bad}
            self.invalid_layouts[self.mode].update(failed)
            with (self.output / "invalid-resets.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "mode": self.mode,
                            "attempt": attempt,
                            "layout_ids": ids,
                            "bad_slots": sorted(bad),
                            "invalid_layouts": sorted(failed),
                        }
                    )
                    + "\n"
                )
            if attempt == 2:
                raise UnStableError("Three invalid reset waves; return to fallback")
            ids, changes = replace_invalid_layouts(
                ids,
                self.invalid_layouts[self.mode],
                len(pool),
                group,
                int(self.cfg["gpu"]) % 2,
            )
            replacements.extend(changes)
        self.totals["reset_seconds"] += time.monotonic() - start
        # EvalEnv.run_eval registers these AFTER reset, which clears all checks.
        # We replace its policy loop only; the native success rule stays intact.
        self.env.run_reward()
        if hasattr(self.env, "get_score"):
            self.env.get_score()
        manager = self.env.reward_manager
        registered = [
            len(manager.check_list[i])
            + len(manager.final_check_list[i])
            + len(manager.trigger_check_list[i])
            for i in range(self.n)
        ]
        if not all(registered):
            raise RuntimeError("Native task has no registered success conditions")
        self.last_obs = self.observations()
        states = self.last_obs["states"].reshape(-1, group, 14)
        gap = float(np.abs(states - states[:, :1]).max())
        audit = {
            "mode": self.mode,
            "layout_ids": ids,
            "group_size": group,
            "requested_layout_ids": requested_ids,
            "physical_reset_replacements": replacements,
            "registered_success_conditions": registered,
            "group_state_max_abs": gap,
            "layouts": [pool[i]["digest"] for i in ids],
        }
        from initial_state import audit_objects

        audit["objects"] = audit_objects(self.env, self.n, group)
        if gap > 0.01:
            raise RuntimeError(f"Group initial robot state mismatch: {gap}")
        save(self.output / "reset-audit.json", audit)
        probe = self.output / "initial-observation.npz"
        if not probe.exists():
            np.savez(probe, **{k: v for k, v in self.last_obs.items() if v is not None})
        self.progress("reset_complete", audit=audit)
        return {"obs": self.last_obs, "audit": audit}

    def step(self, request):
        assert request["mode"] == self.mode, "Mode must be selected by a full reset"
        actions = np.asarray(request["actions"], dtype=np.float32)
        assert (
            actions.ndim == 3 and actions.shape[0] == self.n and actions.shape[2] == 14
        )
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite physical actions")
        horizon = actions.shape[1]
        before = np.asarray(self.env.end_flag[: self.n], dtype=bool).copy()
        dones = np.zeros((self.n, horizon), bool)
        success = np.zeros_like(dones)
        for t in range(horizon):
            active = [i for i in range(self.n) if not self.env.end_flag[i]]
            if active:
                controls = [
                    self.unpack(
                        actions[i, t],
                        "joint",
                        self.env.robot_action_dim_info,
                        source_type="obs",
                    )
                    for i in active
                ]
                start = time.monotonic()
                self.env.take_action_batch(controls, env_idx_list=active)
                if any(i < self.n for i in self.env.unstable_envs):
                    raise RuntimeError(
                        "Physics-invalid rollout: do not score it as policy failure"
                    )
                self.totals["action_seconds"] += time.monotonic() - start
                # Preserve accepted native per-action camera/reward cadence.
                self.last_obs = self.observations()
            dones[:, t] = self.env.end_flag[: self.n]
            success[:, t] = np.asarray(self.env.success[: self.n]) & dones[:, t]
        rewards, terms, truncs = terminal_arrays(before, dones, success)
        self.progress("chunk_complete", successes=int(success[:, -1].sum()))
        return {
            "obs": self.last_obs,
            "rewards": rewards,
            "terminations": terms,
            "truncations": truncs,
            "success": success[:, -1],
            "counts": self.env.take_action_cnt[: self.n],
        }

    def serve(self):
        socket = Path(self.cfg["socket"])
        socket.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = Path(self.cfg["authkey"]).read_bytes()
        with Listener(str(socket), family="AF_UNIX", authkey=key) as listener:
            os.chmod(socket, 0o600)
            self.progress("ready", pools={k: len(v) for k, v in self.pools.items()})
            while True:
                with listener.accept() as connection:
                    try:
                        request = connection.recv()
                        op = request["op"]
                        if op == "describe":
                            value = {
                                "pools": {k: len(v) for k, v in self.pools.items()},
                                "num_envs": self.env.num_envs,
                            }
                        elif op == "reset":
                            value = self.reset(request)
                        elif op == "step":
                            value = self.step(request)
                        elif op == "close":
                            connection.send({"ok": True, "value": None})
                            break
                        else:
                            raise ValueError(f"Unknown op {op}")
                        connection.send({"ok": True, "value": value})
                    except Exception:
                        error = traceback.format_exc()
                        print(error, flush=True)
                        connection.send({"ok": False, "error": error})
        socket.unlink(missing_ok=True)


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    cfg = json.loads(Path(os.environ["DOJO_GRPO_NATIVE_CONFIG"]).read_text())
    root = Path(cfg["repo"])
    sys.path[:0] = [str(root), str(root / "XPolicyLab")]
    # Native main performs AppLauncher before simulator imports, GPU binding,
    # zero-delay rendering, config processing and PhysX monitoring unchanged.
    import src.eval_client.main as native
    import src.eval_client.eval_env as environment

    class NoPolicyClient:
        def __init__(self, **kwargs):
            pass

        def call(self, **kwargs):
            return None

        def close(self):
            pass

    environment.WsModelClient = NoPolicyClient
    factory = native.create_eval_env

    def serve(config, app, **kwargs):
        print("[dojo-grpo] constructing native environments", flush=True)
        env = factory(config, app, **kwargs)
        try:
            NativeLoop(env, cfg).serve()
        finally:
            env.close()
            app.close()
        raise SystemExit(0)

    native.create_eval_env = serve
    native.MAX_INPROC_RESTARTS = 0
    native.main()


if __name__ == "__main__":
    main()
