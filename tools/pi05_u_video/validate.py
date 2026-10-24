"""One CPU pass: U recomputation, full video decode and timeline coverage."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pi05_signals"))
from signal_math import noise_disagreement, ugrow_10_vs_5


def validate(config):
    c = json.loads(Path(config).read_text())
    out = Path(c["output"])
    done = json.loads((out / "done.json").read_text())
    episodes = json.loads((out / "episodes.json").read_text())
    assert len(episodes) == done["episodes"] == 16
    checked = []
    for e in episodes:
        m = json.loads((out / e["frame_map"]).read_text())
        ticks = [f["tick"] for f in m["frames"]]
        assert ticks[0] == 0 and ticks[-1] == m["ticks"] == e["control_steps"]
        assert len(ticks) == e["frames"] and all(0 < b - a <= m["stride"] for a, b in zip(ticks, ticks[1:]))
        cap = cv2.VideoCapture(str(out / e["video"]))
        n = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.size > 0
            n += 1
        cap.release()
        assert n == e["frames"] and n >= 2, (e["slot"], n, e["frames"])
        if c.get("record_observer", False):
            assert m["synchronized_videos"]["observer"] == e["observer_video"]
            cap = cv2.VideoCapture(str(out / e["observer_video"]))
            observer_frames = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                assert image.shape[:2] == (480, 640)
                observer_frames += 1
            cap.release()
            assert observer_frames == n, (e["slot"], observer_frames, n)
        control = [json.loads(line) for line in (out / f"episode_{e['slot']:02d}_control.jsonl").read_text().splitlines()]
        assert len(control) == e["control_steps"]
        assert [r["tick"] for r in control] == list(range(1, len(control) + 1))
        assert e["success"] or e["submitted"] >= c["step_limit"]
        checked.append(dict(slot=e["slot"], frames=n, control_steps=len(control)))
    for q in range(done["queries"]):
        with np.load(out / f"query_{q:03d}.npz") as z:
            score = ugrow_10_vs_5(z["final_model_action"], z["side_action_model"])
            assert score.score.shape == (16, 50)
            assert np.array_equal(score.score, z["u"]) and score.valid.all()
            if c.get("noise_samples") == 4:
                assert z["noise4_model_actions"].shape == (16, 4, 50, 14)
                assert np.array_equal(z["noise4_model_actions"][:, 0], z["final_model_action"])
                assert np.array_equal(z["noise4_env_actions"][:, 0], z["env_action"])
                assert np.array_equal(z["noise4_initial_noise"][:, 0], z["initial_noise_full"])
                dispersion = noise_disagreement(z["noise4_model_actions"])
                assert dispersion.valid.all() and np.array_equal(dispersion.score, z["noise4_u"])
                assert all(not np.array_equal(z["noise4_initial_noise"][:, a], z["noise4_initial_noise"][:, b])
                           for a in range(4) for b in range(a + 1, 4))
    if c["acceptance"]:
        parity = json.loads((out / "u-parity.json").read_text())
        assert parity["action_exact"] and parity["rng_exact"]
        if c.get("noise_samples") == 4:
            parity = json.loads((out / "noise4-parity.json").read_text())
            assert parity["main_action_exact"] and parity["rng_exact"]
    result = dict(time=time.time(), passed=True, video_passed=True, episodes=16,
                  decoded=checked, queries=done["queries"], successes=done["successes"])
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    validate(sys.argv[1])
