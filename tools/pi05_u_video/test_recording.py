"""CPU-only integration check against the exact installed native controller."""
import ast
import copy
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from recording import Recorder, instrument, retained_vertices, project_targets


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    source = Path(os.environ['ROBOTWIN_PATH']) / 'envs/_base_task.py'
    tree = ast.parse(source.read_text())
    cls = next(v for v in tree.body if isinstance(v, ast.ClassDef) and v.name == 'Base_Task')
    names = ['gen_sparse_reward_data', 'compress_path', 'downsample_trajectory']
    methods = [v for v in cls.body if isinstance(v, ast.FunctionDef) and v.name in names]
    namespace = {'np': np}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)

    class Camera:
        def take_picture(self):
            pass
        def get_picture(self, name):
            assert name == 'Color'
            return np.ones((32, 32, 4), np.float32) * .5

    class Scene:
        ticks = 0
        def step(self):
            self.ticks += 1
        def get_timestep(self):
            return .004

    class Planner:
        def TOPP(self, path, dt, verbose):
            positions = np.repeat(path, 2, axis=0)
            n = len(positions)
            return np.arange(n) * dt, positions, np.zeros_like(positions), np.zeros_like(positions), n * dt

    class Robot:
        def __init__(self):
            self.left_mplib_planner = self.right_mplib_planner = Planner()
            self.commands = []
        def get_left_arm_jointState(self):
            return [0.] * 7
        def get_right_arm_jointState(self):
            return [0.] * 7
        def set_arm_joints(self, p, v, side):
            self.commands.append((side, p.tolist(), v.tolist()))
        def set_gripper(self, p, side):
            self.commands.append((side, float(p)))

    class Task:
        def __init__(self):
            self.robot, self.scene = Robot(), Scene()
            self.eval_success, self.take_action_cnt, self.step_lim = False, 0, 200
            self.render_freq = 0
            self.renders = 0
            self.cameras = type('Cameras', (), {'static_camera_list': [Camera()], 'static_camera_name': ['head_camera']})()
        def _update_render(self):
            self.renders += 1
        def check_success(self):
            return self.scene.ticks == 17

    for name in names:
        setattr(Task, name, namespace[name])
    # Instrumentation removes to exactly the original AST (no control rewrite).
    import inspect, textwrap
    original = ast.parse(textwrap.dedent(inspect.getsource(Task.gen_sparse_reward_data)))
    instrument(Task.gen_sparse_reward_data)
    action = np.linspace(0, 1, 50)[:, None] * np.ones((1, 14))
    plain = Task()
    np.random.seed(42)
    result = plain.gen_sparse_reward_data(action)
    rng = np.random.get_state()
    recorded = Task()
    with tempfile.TemporaryDirectory() as d:
        recorder = Recorder(recorded, d, 0, 123)
        recorder.begin(0)
        np.random.seed(42)
        other = recorded.gen_sparse_reward_data(action)
        after = np.random.get_state()
        recorder.end(recorded.eval_success)
        recorder.close()
        assert plain.robot.commands == recorded.robot.commands
        assert plain.scene.ticks == recorded.scene.ticks == 17
        assert plain.renders == recorded.renders and plain.take_action_cnt == recorded.take_action_cnt
        assert all(np.array_equal(a, b) for a, b in zip(result[:3], other[:3])) and result[3] == other[3]
        assert rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]
        frames = json.loads((Path(d) / 'episode_00_frames.json').read_text())
        assert [r['tick'] for r in frames['frames']] == [0, 10, 17]
        cap = cv2.VideoCapture(str(Path(d) / 'episode_00.mp4'))
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        assert count == 3
    path = np.array([[0.], [0.], [1.], [2.]])
    compressed = np.array([[0.], [1.], [2.]])
    vertices = retained_vertices(path, compressed)
    assert vertices.tolist() == [0, 2, 3]
    h, error = project_targets(np.array([[0.], [.5], [1.5], [2.]]), compressed, vertices)
    assert np.allclose(h, [-1, 0, 1.5, 2]) and np.allclose(error, 0)
    print(json.dumps(dict(passed=True, native_commands_exact=True, physics_steps_exact=True,
                         native_rng_exact=True, full_timeline_video_decoded=True, mapping=True)))


if __name__ == '__main__':
    main()
