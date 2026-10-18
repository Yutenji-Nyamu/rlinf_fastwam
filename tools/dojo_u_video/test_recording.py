"""CPU regression: recording preserves native commands, frames and return values."""

import copy
import os
import tempfile
import unittest
from pathlib import Path

from client_entry import patch_env, patch_main, patch_pipeline
from common import read
from recording import instrument


class Writer:
    def __init__(self):
        self.n_frames = 0
        self.width, self.height, self.fps = 640, 480, 25


class FakeEnv:
    def __init__(self):
        self.num_envs, self.task_name, self.eval_seed = 2, "stack_bowls", 0
        self.run_id, self.dt = "cpu", 0.004
        self.take_action_cnt, self.end_flag, self.success = (
            [0, 0],
            [False, False],
            [True, True],
        )
        self.env_seeds = [4, 7]
        self.current_env_seed_map = {0: 4, 1: 7}
        self.video_writers, self.commands, self.ticks = {}, [], 0
        self.save_dir = os.environ["DOJO_U_OUTPUT"]

    def reset(self, seed=None):
        return "unchanged-reset"

    def sim_step(self, render=False):
        self.ticks += 1

    def _stream_vision(self, i, frame):
        writers = self.video_writers.setdefault(
            i, {c: Writer() for c in ["head", "left", "right"]}
        )
        for w in writers.values():
            w.n_frames += 1

    def get_obs_batch(self, env_idx_list=None, last_frame=False):
        ids = [0, 1] if env_idx_list is None else env_idx_list
        for i in ids:
            self._stream_vision(i, {})
        return [dict(env_idx=i, state=[1, 2]) for i in ids]

    def take_action_batch(self, actions_list, env_idx_list=None):
        ids = [0, 1] if env_idx_list is None else env_idx_list
        self.commands.append(copy.deepcopy(actions_list))
        for i in ids:
            self.take_action_cnt[i] += 1
        for _ in range(10):
            self.sim_step(render=False)
        if self.take_action_cnt[0] == 2:
            self.end_flag = [True, True]
            self.get_obs_batch(ids, last_frame=True)
        return "unchanged-action"

    def save_video(self, i, video_path, tag):
        self.video_writers.pop(i)
        return "unchanged-video"


class Tests(unittest.TestCase):
    def test_recording_does_not_change_native_execution(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["DOJO_U_OUTPUT"] = d
            baseline, captured = FakeEnv(), instrument(FakeEnv())
            for env in [baseline, captured]:
                self.assertEqual(env.reset(), "unchanged-reset")
                env.get_obs_batch()
                for n in range(2):
                    self.assertEqual(
                        env.take_action_batch([{"joint": [n]}, {"joint": [n + 1]}]),
                        "unchanged-action",
                    )
                    if n == 0:
                        env.get_obs_batch()
            self.assertEqual(baseline.commands, captured.commands)
            self.assertEqual(baseline.ticks, captured.ticks)
            self.assertEqual(captured.ticks, 20)
            self.assertEqual(
                [w.n_frames for w in baseline.video_writers[0].values()],
                [w.n_frames for w in captured.video_writers[0].values()],
            )
            self.assertEqual(
                captured.save_video(0, d + "/episode.mp4", "success"), "unchanged-video"
            )
            ep = read(Path(d) / "episodes/layout_00004/episode.json")
            self.assertEqual(ep["executed_actions"], 2)
            self.assertEqual(ep["videos"]["head"]["frames"], 3)

    def test_installed_source_anchors(self):
        r = Path(os.environ["DOJO_U_REPO"])
        for relative, patch in [
            ("src/eval_client/main.py", patch_main),
            ("src/eval_client/eval_env.py", patch_env),
            ("utils/pipeline_utils.py", patch_pipeline),
        ]:
            compile(patch((r / relative).read_text()), relative, "exec")

    def test_fixed_budget_and_repeats(self):
        from layouts import make_plan

        with tempfile.TemporaryDirectory() as d:
            for seed in range(3):
                p = Path(d) / str(seed)
                p.mkdir()
                for i in range(30):
                    (p / f"stack_bowls_{i}.json").write_text("{}")
                (p / "stack_bowls_random_0.json").write_text("{}")
            plan = make_plan(d, "stack_bowls", 125)
            self.assertEqual(len(plan), 125)
            self.assertEqual(len({row["scene_layout"] for row in plan}), 90)
            self.assertEqual(sum(row["repeat"] == 1 for row in plan), 35)
            self.assertEqual([row["episode_id"] for row in plan], list(range(125)))

    def test_changed_source_is_rejected(self):
        with self.assertRaises(RuntimeError):
            patch_env("unrecognized source")


if __name__ == "__main__":
    unittest.main()
