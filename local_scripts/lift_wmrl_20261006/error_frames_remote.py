"""Read-only CPU contact sheet for 1 FP, 3 FN, and 2 TP lift_pot test cases.

Returns a small rendered contact sheet as PNG/base64 JSON. It does not export
full videos/arrays, run model inference, change labels, or retune the threshold.
The parent retains media outside Git; metadata can be recorded independently.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import socket

import numpy as np
from PIL import Image, ImageDraw, ImageFont


S = Path("/data/chenyiteng/projects/opendw-robotwin-smoke-20261003")
D = S / "task-reward-v2"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main():
    assert os.getuid() == 20001 and socket.gethostname() == "h100-gpu01"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Read-only media rendering must hide CUDA"
    manifest_path, report_path = D / "data/lift-pot/manifest.json", D / "lift-pot/report.json"
    predictions_path = D / "lift-pot/test_predictions.json"
    report, manifest, predictions = read(report_path), read(manifest_path), read(predictions_path)
    assert report["task_name"] == manifest["task_name"] == "lift_pot"
    assert digest(manifest_path) == report["dataset_manifest_sha256"]
    threshold = float(report["validation"]["threshold"])
    episodes = {row["episode_uid"]: row for row in manifest["episodes"]}
    scores = {(row["episode_uid"], int(row["action_step"])): row for row in predictions}
    selected = []
    for kind in ("FP", "FN", "TP"):
        candidates = [row for row in predictions if
                      (kind == "FP" and row["label"] == 0 and row["score"] >= threshold) or
                      (kind == "FN" and row["label"] == 1 and row["score"] < threshold) or
                      (kind == "TP" and row["label"] == 1 and row["score"] >= threshold)]
        candidates.sort(key=lambda row: (episodes[row["episode_uid"]]["seed"], row["action_step"]))
        if kind == "TP":
            candidates = candidates[:2]
        selected.extend((kind, row) for row in candidates)
    assert [kind for kind, _ in selected].count("FP") == 1
    assert [kind for kind, _ in selected].count("FN") == 3
    assert [kind for kind, _ in selected].count("TP") == 2
    width, column_width, top, row_height = 1000, 324, 64, 318
    sheet = Image.new("RGB", (width, top + row_height * len(selected) + 35), "#f5f7f8")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 9), "lift_pot RM | native held-out test errors and two true positives", fill="#172d39", font=font(19))
    draw.text((12, 37), f"Fixed threshold {threshold:.6f}; center = scored frame; labels after first success may be latched.", fill="#53616b", font=font(13))
    rows = []
    for row_number, (kind, prediction) in enumerate(selected):
        episode = episodes[prediction["episode_uid"]]
        path = Path(episode["source_record"])
        assert path.resolve().is_relative_to(D.resolve())
        assert digest(path) == episode["source_record_sha256"]
        record = read(path)
        assert record["capture_mode"] == "reward_native"
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            frames = archive["native_frames"]
        assert hashlib.sha256(frames.tobytes()).hexdigest() == episode["source_frames_sha256"] == record["native_frames_sha256"]
        center = record["action_steps"].index(int(prediction["action_step"]))
        assert int(record["native_success"][center]) == int(prediction["label"])
        y = top + row_number * row_height
        outcome = "successful episode" if episode["reference_success"] else "failed episode"
        color = "#a53628" if kind in {"FP", "FN"} else "#176960"
        draw.text((12, y), f"{kind} | requested seed {episode['seed']} | score {prediction['score']:.4f} | {outcome}", fill=color, font=font(16))
        row_meta = dict(kind=kind, **prediction, seed=episode["seed"], reference_success=episode["reference_success"],
                        source_record=str(path), source_record_sha256=episode["source_record_sha256"], frames=[])
        for column, index in enumerate((center - 1, center, center + 1)):
            x, image_y = 12 + column * column_width, y + 52
            if not 0 <= index < len(frames):
                draw.text((x, y + 30), "No adjacent frame", fill="#596873", font=font(13))
                continue
            action_step, native_label = int(record["action_steps"][index]), record["native_success"][index]
            first = record["first_success_position"]
            label_text = "unknown" if native_label is None else str(int(native_label))
            latched = first is not None and index > first
            if latched:
                label_text += " (latched)"
            entry = scores.get((prediction["episode_uid"], action_step))
            score_text = f"p={entry['score']:.3f}" if entry else "not scored"
            draw.text((x, y + 29), f"step {action_step} | native={label_text} | {score_text}", fill="#344853", font=font(12))
            image = Image.fromarray(frames[index])
            image.thumbnail((316, 244), Image.Resampling.BILINEAR)
            sheet.paste(image, (x, image_y))
            if column == 1:
                draw.rectangle((x - 1, image_y - 1, x + image.width, image_y + image.height), outline=color, width=2)
            row_meta["frames"].append(dict(action_step=action_step, native_label=native_label,
                                            post_first_success_latched=latched,
                                            score=None if entry is None else float(entry["score"]),
                                            original_shape=list(frames[index].shape), is_selected=index == center))
        rows.append(row_meta)
    draw.text((12, sheet.height - 27), "Visual diagnostic only. Simulator labels remain unchanged; this is not an OpenDW-domain accuracy measurement.",
              fill="#53616b", font=font(12))
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=True)
    pixels = buffer.getvalue()
    print(json.dumps(dict(schema_version=1, task="lift_pot", threshold=threshold,
                          report_sha256=digest(report_path), test_predictions_sha256=digest(predictions_path),
                          selection="all test FP/FN; first two TP sorted by requested seed", rows=rows,
                          media=dict(mime_type="image/png", width=sheet.width, height=sheet.height,
                                     bytes=len(pixels), sha256=hashlib.sha256(pixels).hexdigest(),
                                     base64=base64.b64encode(pixels).decode("ascii")))))


if __name__ == "__main__":
    main()
