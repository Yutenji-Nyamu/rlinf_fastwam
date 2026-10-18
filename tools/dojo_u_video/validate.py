"""Validate completed native episodes without rereading large weights."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from common import read, save


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def validate(output, full_decode=False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi05_signals"))
    from signal_math import ugrow_10_vs_5

    output = Path(output)
    rows = []
    for path in sorted(output.glob("episodes/*/episode.json")):
        ep = read(path)
        directory = path.parent
        actions = lines(directory / "actions.jsonl")
        frames = lines(directory / "frames.jsonl")
        assert [a["action_index"] for a in actions] == list(
            range(ep["executed_actions"])
        )
        assert len(ep["videos"]) == 3, "Incomplete native camera coverage"
        queries = {}
        for a in actions:
            q = a["query"]
            if q not in queries:
                with np.load(directory / f"query_{q:04d}.npz") as data:
                    u = ugrow_10_vs_5(
                        data["main_endpoint"][None, :, :14],
                        data["side_endpoint"][None, :, :14],
                    )
                    assert np.allclose(data["u"], u.score[0], rtol=0, atol=1e-12)
                    assert (
                        data["initial_noise_full"].shape == data["main_endpoint"].shape
                    )
                    queries[q] = data["u"].copy()
                assert (directory / f"query_{q:04d}_obs.npz").is_file()
            assert 0 <= a["h"] < 50 and a["end_tick"] > a["start_tick"]
        for cam, v in ep["videos"].items():
            fs = [f for f in frames if f["camera"] == cam]
            assert [f["frame"] for f in fs] == list(range(v["frames"]))
            assert fs[0]["action_before"] == 0
            assert (
                fs[-1]["action_before"] == ep["executed_actions"] and fs[-1]["terminal"]
            )
            assert all(
                b["control_tick"] >= a["control_tick"] for a, b in zip(fs, fs[1:])
            )
            assert len(fs) >= ep["executed_actions"] + 1
            p = Path(v["path"])
            assert p.is_file() and p.stat().st_size > 1000
            # Read actual stream length, not only the writer's submitted count.
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames",
                    "-of",
                    "csv=p=0",
                    str(p),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            assert int(probe.stdout.strip()) == v["frames"]
            if full_decode and not rows:
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    timeout=180,
                )
        rows.append(
            dict(
                task=ep["task"],
                layout=ep["layout"],
                success=ep["success"],
                actions=len(actions),
                queries=len(queries),
                path=str(path),
                peak_u=max(float(x.max()) for x in queries.values()),
                u_range=max(float(x.max()) for x in queries.values())
                - min(float(x.min()) for x in queries.values()),
            )
        )
    if not rows:
        raise ValueError("No completed episodes with video and U")
    result = dict(
        passed=True,
        full_decode_first_episode=full_decode,
        episodes=len(rows),
        successes=sum(r["success"] for r in rows),
        rows=rows,
    )
    save(output / "validation.json", result)
    save(
        output / "candidates.json",
        sorted([r for r in rows if r["success"]], key=lambda r: -r["u_range"])[:12],
    )
    return result


if __name__ == "__main__":
    print(json.dumps(validate(sys.argv[1], "--full-decode" in sys.argv)))
