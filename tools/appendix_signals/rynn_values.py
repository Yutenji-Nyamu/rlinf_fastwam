"""Causal prefix-only RynnValue scores, with fixed pools and explicit units."""

import argparse
import json
from pathlib import Path

import numpy as np

from .common import (
    Timer,
    causal_prefix,
    cuda_guard,
    sha256,
    uniform_indices,
    write_json,
    write_npz,
)


def build_pool(data):
    rows = json.loads((data / "rt-pool/queries.json").read_text())
    pool = []
    for task in ["turn_switch", "stack_bowls_two"]:
        episodes = []
        for row in rows:
            if row["task"] == task and row["slot"] not in [x["slot"] for x in episodes]:
                episodes.append(row)
        for row in episodes[:6]:
            ep = json.loads((Path(row["batch"]) / "episodes.json").read_text())[
                row["slot"]
            ]
            pool.append(
                dict(
                    id=f"rt2_{task}_{row['slot']:02d}",
                    bench="rt2",
                    task=task,
                    slot=row["slot"],
                    success=row["success"],
                    instruction=row["instruction"],
                    video=str(Path(row["batch"]) / ep["video"]),
                    frame_map=str(Path(row["batch"]) / ep["frame_map"]),
                    robot="a dual-arm ALOHA robot",
                    camera="a fixed front-facing head camera",
                )
            )
    for task in ["put_bottles_into_dustbin", "pour_liquid_into_cup"]:
        d = data / "dojo" / task
        if not (d / "complete.json").exists():
            continue
        episodes = json.loads((d / "episodes.json").read_text())
        yes = [e for e in episodes if e["success"]]
        no = [e for e in episodes if not e["success"]]
        selected = yes[:2] + no[:2]
        selected += [e for e in episodes if e not in selected][: 4 - len(selected)]
        for e in selected:
            with np.load(d / "observations/q000.npz", allow_pickle=False) as z:
                instruction = str(z["task_descriptions"][e["slot"]])
            pool.append(
                dict(
                    id=f"dojo_{task}_{e['slot']:02d}",
                    bench="dojo",
                    task=task,
                    slot=e["slot"],
                    success=e["success"],
                    instruction=instruction,
                    directory=str(d),
                    robot="a dual-arm ARX X5 robot",
                    camera="a fixed front-facing head camera",
                )
            )
    assert len(pool) <= 20
    for task, wrong in [
        ("turn_switch", "Stack two bowls on top of one another."),
        ("stack_bowls_two", "Press the switch button."),
    ]:
        candidates = [x for x in pool if x["bench"] == "rt2" and x["task"] == task]
        for ep in [candidates[0], candidates[-1]]:
            altered = dict(ep)
            altered.update(
                id=ep["id"] + "_mismatch",
                counterfactual_of=ep["id"],
                correct_instruction=ep["instruction"],
                instruction=wrong,
                instruction_match=False,
                anchor_cap=8,
            )
            pool.append(altered)
    assert len(pool) <= 24
    return pool


def images_for(ep):
    from PIL import Image

    if ep["bench"] == "rt2":
        import cv2

        cap = cv2.VideoCapture(ep["video"])
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        mapping = json.loads(Path(ep["frame_map"]).read_text())["frames"]
        assert len(frames) == len(mapping)
        coordinates = np.asarray([f["sim_seconds"] for f in mapping])
        unit = "simulation seconds"
    else:
        d = Path(ep["directory"])
        s = ep["slot"]
        with np.load(d / "observations/q000.npz", allow_pickle=False) as z:
            frames = [Image.fromarray(z["main_images"][s])]
        coordinates = [0]
        for p in sorted((d / "frames").glob("after_q*.npz")):
            with np.load(p, allow_pickle=False) as z:
                if not z["active_before"][s]:
                    break
                frames.append(Image.fromarray(z["main_images"][s]))
                coordinates.append(int(z["native_action_steps"][s]))
                if z["done"][s]:
                    break
        coordinates = np.asarray(coordinates)
        unit = "native action steps"
    assert frames and np.all(np.diff(coordinates) >= 0)
    # Match the official adapter's maximum 640px; existing head frames are smaller.
    resized = []
    for f in frames:
        if max(f.size) > 640:
            scale = 640 / max(f.size)
            f = f.resize(
                tuple(round(s * scale) for s in f.size), Image.Resampling.BICUBIC
            )
        resized.append(f)
    return resized, coordinates, unit


def main(args):
    torch = cuda_guard()
    from transformers import AutoConfig, AutoModel, AutoProcessor

    data = Path(args.data)
    out = data / "rynn"
    out.mkdir(parents=True, exist_ok=True)
    pool = build_pool(data)
    # Persist before model inference; do not select examples by their scores.
    write_json(out / "candidate-pool.json", pool)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    model = (
        AutoModel.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    assert int(config.num_value_heads) == 1, (
        "This checkpoint must have one absolute-value head"
    )
    write_json(
        out / "model-contract.json",
        dict(
            attention="sdpa causal, final slot only",
            heads=1,
            prefix_frames=8,
            units="remaining seconds; no success-probability interpretation",
            configuration_sha256=sha256(Path(args.model) / "config.json"),
            code_sha256=sha256(Path(args.model) / "modeling_rynn_value_lang.py"),
            source="official RynnValue train-isolated/infer-causal recommendation; RoboValue prefix sampling convention",
        ),
    )
    receipts = []
    for ep in pool:
        target = out / ep["id"]
        target.mkdir(exist_ok=True)
        images, coordinates, unit = images_for(ep)
        anchors = uniform_indices(len(images), ep.get("anchor_cap", 16))
        index = np.stack([causal_prefix(int(a), 8) for a in anchors])
        assert np.all(index <= anchors[:, None])
        samples = [
            processor.process_episode(
                instruction=ep["instruction"],
                images=[images[j] for j in ids],
                robot_description=ep["robot"],
                camera_description=ep["camera"],
            )
            for ids in index
        ]
        raw_values = []
        timings = []
        for lo in range(0, len(samples), 8):
            chunk = samples[lo : lo + 8]
            # Within one episode the instruction and image sizes are identical.
            assert len({x["input_ids"].shape[-1] for x in chunk}) == 1
            batch = dict(
                input_ids=torch.cat([s["input_ids"] for s in chunk]).cuda().long(),
                attention_mask=torch.cat([s["attention_mask"] for s in chunk])
                .cuda()
                .long(),
                pixel_values=torch.cat(
                    [s["pixel_values"].flatten(0, 1) for s in chunk]
                ).cuda(),
                image_grid_thw=torch.cat(
                    [s["image_grid_thw"].flatten(0, 1) for s in chunk]
                )
                .cuda()
                .long(),
            )
            with Timer() as timer, torch.inference_mode():
                result = model(**batch)
            raw = result.value.pred_value.detach().float().cpu().numpy()
            assert raw.size == len(chunk) * 8 and np.isfinite(raw).all(), raw.shape
            raw_values.append(raw.reshape(len(chunk), 8))
            timings.append(
                dict(batch=len(chunk), raw_shape=list(raw.shape), **timer.result)
            )
        values = np.concatenate(raw_values)
        write_npz(
            target / "values.npz",
            all_prefix_values=values,
            remaining_seconds=values[:, -1],
            prefix_indices=index,
            anchor_indices=anchors,
            anchor_coordinate=coordinates[anchors],
            all_frame_coordinates=coordinates,
        )
        for i in uniform_indices(len(anchors), 4):
            images[int(anchors[i])].save(
                target / f"anchor_{int(i):02d}.jpg", quality=90
            )
        receipt = dict(
            **ep,
            coordinate_unit=unit,
            queries=len(anchors),
            timings=timings,
            source_frame_count=len(images),
        )
        write_json(target / "complete.json", receipt)
        receipts.append(receipt)
        write_json(out / "index.json", receipts)
    write_json(
        out / "complete.json",
        dict(
            status="complete",
            episodes=len(receipts),
            queries=sum(x["queries"] for x in receipts),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    main(args)
