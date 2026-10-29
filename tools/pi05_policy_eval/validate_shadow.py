"""Validate raw paired signals and complete-video endpoints; write light curves."""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from teacher_shadow import compare


def validate(config):
    c = json.loads(Path(config).read_text())
    out = Path(c["output"])
    done = json.loads((out / "done.json").read_text())
    episodes = json.loads((out / "episodes.json").read_text())
    assert done["teacher_shadow"] and len(episodes) == done["episodes"] == c["num_envs"]
    assert json.loads((out / "shadow-parity.json").read_text())["rng_exact"]
    frames = []
    rows = []
    for e in episodes:
        m = json.loads((out / e["frame_map"]).read_text())
        ticks = [r["tick"] for r in m["frames"]]
        assert ticks[0] == 0 and ticks[-1] == m["ticks"] == e["control_steps"]
        assert len(ticks) == e["frames"] and all(
            0 < b - a <= m["stride"] for a, b in zip(ticks, ticks[1:])
        )
        cap = cv2.VideoCapture(str(out / e["video"]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert n == e["frames"] and n >= 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        ok, img = cap.read()
        cap.release()
        assert ok and img.size
        frames.append({"slot": e["slot"], "frames": n, "terminal_frame_readable": True})
    for query in range(done["queries"]):
        with np.load(out / f"query_{query:03d}.npz") as z:
            assert z["noise4_model_actions"].shape == (c["num_envs"], 4, 50, 14)
            assert np.array_equal(
                z["noise4_initial_noise"][:, 0], z["teacher_initial_noise"]
            )
            assert np.array_equal(
                z["noise4_model_actions"][:, 0], z["final_model_action"]
            )
            assert not z["selected"].any()
            recalculated = compare(
                z["noise4_model_actions"], z["teacher_model_actions"]
            )
            for key, value in recalculated.items():
                np.testing.assert_array_equal(value, z[key])
            for slot in np.flatnonzero(z["active"]):
                executed = int(z["after"][slot] - z["before"][slot])
                assert 0 <= executed <= c["execute_chunk"]
                for h in range(executed):
                    rows.append(
                        {
                            "slot": int(slot),
                            "seed": episodes[slot]["actual"],
                            "query": query,
                            "action": int(z["before"][slot]) + h,
                            "success": episodes[slot]["success"],
                            "u": float(z["u"][slot, h]),
                            "noise4_u": float(z["noise4_u"][slot, h]),
                            "noise3_u": float(z["noise3_without_main_u"][slot, h]),
                            "teacher_student_rmse": float(
                                z["teacher_student_rmse"][slot, h]
                            ),
                        }
                    )
        assert (out / f"observation_{query:03d}.npz").is_file()
    with (out / "signals.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "passed": True,
        "episodes": len(episodes),
        "queries": done["queries"],
        "executed_actions": len(rows),
        "successes": done["successes"],
        "frames": frames,
        "video_validation": "all metadata coverage and decoded terminal frames; accepted unchanged recorder",
        "metric": "noise4 normalized dispersion; same-noise teacher/student normalized RMSE; not probability entropy",
    }
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    validate(sys.argv[1])
