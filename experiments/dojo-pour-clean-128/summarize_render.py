"""Summarize persisted native events, including a partial crossover."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent / "trial"
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def rms(a, b):
    return float(np.mean(np.square(a.astype(np.float32)-b.astype(np.float32)))**0.5)

def lane(card):
    directory = root / f"g{card}"
    completed = rows(directory / "completed-waves.jsonl")[0]["rows"]
    events = rows(directory / "timing.jsonl")
    timing = next(event for event in events if event["event"] == "wave_complete")
    resets = [e for e in events if e["event"] == "reset_attempt"]
    reset = json.loads((directory / "reset-audit.json").read_text())
    return dict(episodes=completed, action_seconds=timing["action_seconds"],
                observation_seconds=timing["observation_seconds"], first_reset_seconds=resets[0]["seconds"],
                first_reset_invalid_slots=resets[0]["invalid_slots"],
                last_reset_layouts=reset["layout_ids"], last_reset_replacements=reset["physical_reset_replacements"])

standard, boundary = lane(5), lane(6)
with np.load(root/"g5/initial-observation.npz") as a, np.load(root/"g6/initial-observation.npz") as b:
    initial = []
    for i, item in enumerate(standard["episodes"]):
        initial.append(dict(layout=item["layout"], state_rms=rms(a["states"][i],b["states"][i]),
            main_rgb_rms=rms(a["main_images"][i],b["main_images"][i]),
            wrist0_rgb_rms=rms(a["wrist_images"][i,0],b["wrist_images"][i,0]),
            wrist1_rgb_rms=rms(a["wrist_images"][i,1],b["wrist_images"][i,1])))
last = json.loads((root/"progress.json").read_text())
assert last["run"]==0 and last["chunk"]==8
seconds = last["seconds"]
out = dict(layouts=[e["layout"] for e in standard["episodes"]], first_pass_step_wall_seconds=seconds,
           wall_fraction_saved=1-seconds[1]/seconds[0], initial_observation_rmse=initial,
           reference=standard, query_and_terminal=boundary,
           first_pass_seeds_matched=[e["layout"] for e in standard["episodes"]]==[e["layout"] for e in boundary["episodes"]],
           crossed_pass_completed=False, final_train_render="native_per_action",
           decision="Two episode outcomes differed on simultaneous fixed-action replay. Native reset substituted layout 32 for 0 on the crossed pass. The effect of rendering cannot be isolated from simulation variability.")
(root/"observed-render.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out))

