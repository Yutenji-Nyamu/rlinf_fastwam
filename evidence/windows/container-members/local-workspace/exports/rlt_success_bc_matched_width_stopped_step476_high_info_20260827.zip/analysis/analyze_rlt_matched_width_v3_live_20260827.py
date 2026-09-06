from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/rlt_success_bc_matched_width_live_20260827"
HISTORICAL = [
    ROOT / "exports/rlt_stage2_formal_8env250_high_info_20260730_v1/runtime/driver.log",
    ROOT / "exports/rlt_stage2_formal_resume250_to480_high_info_20260731_v1/driver.log",
]
CONTROL = ROOT / "tmp/rlt_v3_control_metrics_live.log"
METHOD = ROOT / "tmp/rlt_v3_method_metrics_live.log"

INK = "#172033"
GRID = "#D9E1EA"
HIST = "#4B5563"
CONTROL_COLOR = "#00796B"
METHOD_COLOR = "#E66101"
BG = "#F7F9FC"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def extract(section: str, pattern: str):
    m = re.search(pattern, section)
    return float(m.group(1)) if m else None


def parse(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(re.finditer(r"Global Step:\s*(\d+)/\d+", text))
    rows = []
    for i, m in enumerate(matches):
        block = text[m.start() : matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        a = block.find(" Environment ")
        if a < 0:
            continue
        b = block.find(" Evaluation ", a)
        c = block.find(" Replay Buffer ", a)
        stop = b if b >= 0 else c
        env = block[a : stop if stop >= 0 else len(block)]
        success = extract(env, r"success_once=([-+0-9.eE]+)")
        if success is None:
            continue
        row = {"step": int(m.group(1)), "success": success * 100}
        if b >= 0:
            ev = block[b : c if c >= 0 else len(block)]
            value = extract(ev, r"success_once=([-+0-9.eE]+)")
            if value is not None:
                row["eval"] = value * 100
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("step", keep="last").sort_values("step")


def load(paths):
    if isinstance(paths, Path):
        return parse(paths)
    return pd.concat([parse(p) for p in paths], ignore_index=True).drop_duplicates("step", keep="last").sort_values("step")


def rolling(frame):
    frame = frame.copy()
    frame["ma5"] = frame.success.rolling(5, min_periods=1).mean()
    frame["ma10"] = frame.success.rolling(10, min_periods=1).mean()
    return frame


def plot(frames, latest):
    width, height = 1900, 1500
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((60, 45), f"RLT training success through common Step {latest}", fill=INK, font=font(42, True))
    draw.text((60, 102), "Historical two-GPU RLT is context; current control vs DVAC-BC is the matched A/B.", fill="#52627A", font=font(23))

    boxes = [(105, 205, 910, 755), (1015, 205, 1820, 755), (105, 865, 910, 1415), (1015, 865, 1820, 1415)]
    specs = [("Raw train success", "success"), ("Trailing 5-step mean", "ma5"), ("Trailing 10-step mean", "ma10"), ("Fixed-20 evaluation", "eval")]

    def xy(step, value, box):
        l, t, r, b = box
        x = l + (step - 1) / max(1, latest - 1) * (r - l)
        y = b - value / 100 * (b - t)
        return round(x), round(y)

    colors = {"historical": HIST, "control": CONTROL_COLOR, "method": METHOD_COLOR}
    dashes = {"historical": True, "control": False, "method": False}
    labels = {"historical": "Historical RLT", "control": "Current control", "method": "DVAC-BC"}

    for box, (title, key) in zip(boxes, specs):
        l, t, r, b = box
        draw.rounded_rectangle((l - 55, t - 70, r + 45, b + 55), radius=18, fill=WHITE, outline="#D7E0EA", width=2)
        draw.text((l - 25, t - 52), title, fill=INK, font=font(28, True))
        for tick in (0, 25, 50, 75, 100):
            y = xy(1, tick, box)[1]
            draw.line((l, y, r, y), fill=GRID, width=1)
            draw.text((l - 14, y), str(tick), fill="#64748B", font=font(18), anchor="rm")
        for tick in range(50, latest + 1, 50):
            x = xy(tick, 0, box)[0]
            draw.line((x, b, x, b + 7), fill="#64748B", width=1)
            draw.text((x, b + 13), str(tick), fill="#64748B", font=font(17), anchor="ma")
        if 126 <= latest:
            x = xy(126, 0, box)[0]
            draw.line((x, t, x, b), fill="#8B5CF6", width=2)

        for name, frame in frames.items():
            data = frame.dropna(subset=[key]) if key in frame else frame.iloc[0:0]
            pts = [xy(float(row.step), float(row[key]), box) for _, row in data.iterrows()]
            if key == "eval":
                if len(pts) > 1:
                    draw.line(pts, fill=colors[name], width=4)
                for point in pts:
                    draw.ellipse((point[0]-6, point[1]-6, point[0]+6, point[1]+6), fill=WHITE, outline=colors[name], width=3)
            elif len(pts) > 1:
                if dashes[name]:
                    for i in range(0, len(pts)-1, 2):
                        draw.line((pts[i], pts[i+1]), fill=colors[name], width=3)
                else:
                    draw.line(pts, fill=colors[name], width=4)

        for j, name in enumerate(("historical", "control", "method")):
            x0 = l + 320 + (j % 2) * 250
            y0 = t - 45 + (j // 2) * 25
            draw.line((x0, y0, x0 + 35, y0), fill=colors[name], width=4)
            draw.text((x0 + 45, y0), labels[name], fill="#475569", font=font(17), anchor="lm")

    draw.text((1045, 1390), "Purple line: current 20k warmup boundary / DVAC apply begins", fill="#6D28D9", font=font(18))
    path = OUT / f"RLT_HISTORICAL_CONTROL_DVAC_BC_THROUGH_G{latest}.png"
    image.save(path)
    return path


def stats(frame, start=1, end=None):
    x = frame[(frame.step >= start) & (frame.step <= (end or frame.step.max()))]
    return {
        "n": int(len(x)),
        "mean": float(x.success.mean()),
        "last5": float(x.success.tail(5).mean()),
        "last10": float(x.success.tail(10).mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {
        "historical": rolling(load(HISTORICAL)),
        "control": rolling(load(CONTROL)),
        "method": rolling(load(METHOD)),
    }
    latest = min(int(frames["control"].step.max()), int(frames["method"].step.max()))
    for key in frames:
        frames[key] = frames[key][frames[key].step <= latest].copy()
        frames[key].insert(0, "run", key)
    combined = pd.concat(frames.values(), ignore_index=True)
    combined.to_csv(OUT / "success_curves.csv", index=False)
    path = plot(frames, latest)
    summary = {
        "common_step": latest,
        "through_common_step": {k: stats(v, end=latest) for k, v in frames.items()},
        "post_apply_step126": {k: stats(v, start=126, end=latest) for k, v in frames.items()},
        "fixed20": {k: [[int(r.step), int(round(r.eval / 5)), float(r.eval)] for _, r in v.dropna(subset=["eval"]).iterrows()] for k, v in frames.items()},
        "plot": path.name,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
