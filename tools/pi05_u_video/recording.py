"""Observe the native controller without changing its actions, timing or RNG.

Video samples the full simulated timeline at 25 Hz, plus both boundaries.
Every physics step is logged. TOPP target-to-model-h coordinates are labelled
as geometric projections; raw paths/targets are retained for re-analysis.
"""
import inspect
import json
import textwrap
import time
import types
from pathlib import Path

import numpy as np


def instrument(original):
    source = textwrap.dedent(inspect.getsource(original))
    inserts = {
        "    steps_executed = 0\n":
            "    steps_executed = 0\n    self._u_video.plan(self, locals())\n",
        "        steps_executed += 1\n":
            "        steps_executed += 1\n        self._u_video.step(self, locals())\n",
    }
    for anchor, replacement in inserts.items():
        if source.count(anchor) != 1:
            raise RuntimeError("Native controller changed; recording insertion rejected")
        source = source.replace(anchor, replacement)
    namespace = {}
    exec(compile(source, "<native-controller-u-video>", "exec"), original.__globals__, namespace)
    return namespace[original.__name__]


def retained_vertices(original, compressed):
    """Exact compression vertex identities; synthetic fallback is marked -1."""
    result = []
    start = 0
    for vertex in compressed:
        found = next((i for i in range(start, len(original))
                      if np.array_equal(original[i], vertex)), None)
        result.append(-1 if found is None else found)
        if found is not None:
            start = found + 1
    return np.asarray(result, dtype=np.int32)


def project_targets(targets, path, vertices):
    """Monotone segment projection, not an exact TOPP inverse parameter."""
    positions, residuals = [], []
    floor = 0
    for point in targets:
        delta = np.diff(path, axis=0)
        denominator = np.sum(delta * delta, axis=1)
        t = np.clip(np.sum((point - path[:-1]) * delta, axis=1)
                    / np.maximum(denominator, 1e-24), 0, 1)
        errors = np.linalg.norm(path[:-1] + t[:, None] * delta - point, axis=1)
        errors[:floor] = np.inf
        segment = int(np.argmin(errors))
        floor = segment
        a, b = vertices[segment:segment + 2]
        positions.append(float(a + t[segment] * (b - a) - 1)
                         if a >= 0 and b >= 0 else np.nan)
        residuals.append(float(errors[segment]))
    return np.asarray(positions), np.asarray(residuals)


class Recorder:
    def __init__(self, task, output, slot, seed, fps=25):
        import cv2
        self.cv2 = cv2
        self.task, self.slot, self.seed = task, slot, seed
        self.output = Path(output)
        self.dt = float(task.scene.get_timestep())
        self.stride = max(1, round(1 / (fps * self.dt)))
        self.fps = 1 / (self.stride * self.dt)
        cameras = task.cameras
        self.camera = cameras.static_camera_list[cameras.static_camera_name.index("head_camera")]
        rgb = self.rgb(take=False)
        self.writer = cv2.VideoWriter(str(self.output / f"episode_{slot:02d}.mp4"),
                                      cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                      (rgb.shape[1], rgb.shape[0]))
        if not self.writer.isOpened():
            raise RuntimeError("Cannot create complete-timeline video")
        self.log = (self.output / f"episode_{slot:02d}_control.jsonl").open("x")
        self.frames = []
        self.query, self.ticks, self.query_start = -1, 0, 0
        self.last_tick = -1
        self.seconds = 0.0
        self.state = {}
        self.capture(rgb, "initial")
        task._u_video = self
        task.gen_sparse_reward_data = types.MethodType(instrument(task.gen_sparse_reward_data.__func__), task)

    def rgb(self, take=True):
        if take:
            self.camera.take_picture()
        return (self.camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype("uint8")

    def begin(self, query):
        self.query = query
        self.query_start = self.ticks
        self.state = {}

    def plan(self, task, scope):
        arrays = {"input_actions": scope["actions"]}
        for side in ("left", "right"):
            original = np.vstack((scope[side + "_current_qpos"], scope[side + "_arm_actions"]))
            path = scope[side + "_path"]
            vertices = retained_vertices(original, path)
            valid = bool(scope["topp_" + side + "_flag"])
            target = scope[side + "_result"]["position"] if valid else np.empty((0, original.shape[1]))
            h, error = project_targets(target, path, vertices)
            arrays.update({side + "_original_path": original, side + "_compressed_path": path,
                           side + "_retained_indices": vertices, side + "_targets": target,
                           side + "_h_projection": h, side + "_projection_error": error,
                           side + "_gripper": scope[side + "_gripper"],
                           side + "_topp_valid": np.asarray(valid)})
        self.mapping = arrays
        np.savez_compressed(self.output / f"q{self.query:03d}_s{self.slot:02d}_controller.npz", **arrays)

    def step(self, task, scope):
        self.ticks += 1
        row = {"query": self.query, "tick": self.ticks,
               "query_tick": int(scope["steps_executed"]), "sim_seconds": self.ticks * self.dt}
        for side in ("left", "right"):
            index = int(scope["now_" + side + "_id"]) - 1
            row[side + "_target_index"] = index
            h = self.mapping[side + "_h_projection"]
            value = float(h[index]) if 0 <= index < len(h) else float("nan")
            row[side + "_h_projection"] = value if np.isfinite(value) else None
        self.state = row
        self.log.write(json.dumps(row) + "\n")
        if self.ticks % self.stride == 0:
            self.capture(self.rgb(), "periodic")

    def capture(self, rgb, reason):
        tic = time.perf_counter()
        self.writer.write(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))
        self.frames.append(dict(self.state, frame=len(self.frames), slot=self.slot,
                                seed=self.seed, query=self.query, tick=self.ticks,
                                sim_seconds=self.ticks * self.dt, reason=reason,
                                after_physics_step=self.ticks > 0))
        self.last_tick = self.ticks
        self.seconds += time.perf_counter() - tic

    def end(self, success):
        if self.ticks != self.last_tick:
            self.capture(self.rgb(take=False), "query_end")
        self.frames[-1]["success_after_query"] = bool(success)
        self.log.flush()

    def close(self):
        self.writer.release()
        self.log.close()
        data = {"slot": self.slot, "seed": self.seed, "fps": self.fps,
                "physics_dt": self.dt, "stride": self.stride,
                "ticks": self.ticks, "frames": self.frames,
                "h_mapping": "monotone geometric projection onto compressed arm path; approximate, per arm",
                "coverage": "initial, every stride physics steps, each query end including success"}
        (self.output / f"episode_{self.slot:02d}_frames.json").write_text(json.dumps(data))
